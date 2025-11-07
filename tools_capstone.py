import json
import requests
from datetime import datetime
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

## --------- Define API Key Variables ----- ##
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]
ZOYA_API_KEY = st.secrets["ZOYA_API_KEY"]

## -------- RETRIEVE FOR FINNHUB API ---------##
def retrieve_from_endpoint(url: str) -> dict:

    """Fungsi helper untuk melakukan GET request ke FINNHUB API."""

    # Header otentikasi untuk setiap request
    headers = {"X-Finnhub-Token": FINNHUB_API_KEY}

    try:
        # Melakukan request GET ke API
        response = requests.get(url, headers=headers)

        # Jika status bukan 200 (OK), langsung raise error
        response.raise_for_status()

        # Mengubah respons JSON menjadi dict Python
        data = response.json()

        return data

    except requests.exceptions.HTTPError as err:
        # Menangani error HTTP
        return {
            "error": f"HTTPError {err.response.status_code} - {err.response.reason}",
            "url": url,
            "detail": err.response.text
        }
    
    except Exception as e:
        # Menangani error tak terduga
        return {
            "error": f"Unexpected error: {type(e).__name__} - {str(e)}",
            "url": url
        }
    
## ------ RETRIEVE FOR ZOYA API -------- ##
def retrieve_graphql_endpoint(
    endpoint: str,
    query: str,
    variables: dict | None = None,
    operation_name: str | None = None,
    extra_headers: dict | None = None,
) -> dict:
    """
    A robust, reusable helper function to perform GraphQL POST requests.

    Args:
        endpoint (str): GraphQL endpoint URL 
        query (str): GraphQL document (query/mutation)
        variables (dict|None): Variables for the operation
        operation_name (str|None): Operation name, if the document has multiple ops
        extra_headers (dict|None): Additional headers to include

    Returns:
        dict: On success -> {"data": {...}}
              On GraphQL error -> {"error": "GraphQLError", "errors": [...], "request": {...}}
              On HTTP/other error -> same style as your REST helper
    """
    headers = {
        "Authorization": ZOYA_API_KEY,         
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload = {"query": query}
    if variables is not None:
        payload["variables"] = variables
    if operation_name:
        payload["operationName"] = operation_name

    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        body = response.json()

        # GraphQL can return 200 with "errors"
        if isinstance(body, dict) and body.get("errors"):
            return {
                "error": "GraphQLError",
                "errors": body["errors"],
                "request": {
                    "endpoint": endpoint,
                    "operationName": operation_name,
                    "variables": variables,
                },
            }

        return {"data": body.get("data") if isinstance(body, dict) else None}

    except requests.exceptions.HTTPError as err:
        return {
            "error": f"HTTPError {err.response.status_code} - {err.response.reason}",
            "url": endpoint,
            "detail": err.response.text,
        }
    except ValueError as e:  # JSON decode error, etc.
        return {
            "error": f"Unexpected error: {type(e).__name__} - {str(e)}",
            "url": endpoint,
            "detail": "Failed to parse JSON response.",
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {type(e).__name__} - {str(e)}",
            "url": endpoint,
        }

## -------- Tools untuk Get Compliance Report from Zoya API --------- ##
query_compliance_report = """
query getReport($symbol: String!) {
  basicCompliance {
    report(symbol: $symbol) {
      exchange
      name
      purificationRatio
      reportDate
      status
      symbol
    }
  }
}
"""
## ----- Tools untuk mengambil compliance report ---- ##
@tool
def get_compliance_report(symbol: str) -> dict:
    """
    Retrieve the basic compliance report for a given stock symbol using a GraphQL API.

    Args:
        symbol (str): Ticker symbol 
        endpoint_override (str|None): Optional GraphQL endpoint to override env (ZOYA_API_KEY).

    Returns:
        dict:
            - On success: {"data": {"basicCompliance": {"report": {...}}}}
            - On GraphQL error: {"error": "GraphQLError", "errors": [...], "request": {...}}
            - On HTTP/other error: {"error": "...", "url": "...", "detail": "..."}  # mirrors retrieve_graphql

    """
    endpoint = "https://api.zoya.finance/graphql"

    # Light input sanitation
    if not isinstance(symbol, str) or not symbol.strip():
        return {"error": "InvalidInput", "detail": "symbol must be a non-empty string."}

    variables = {"symbol": symbol.strip().upper()}

    # Delegate the network call + error handling to your simple helper
    result = retrieve_graphql_endpoint(
        endpoint=endpoint,
        query=query_compliance_report,
        variables=variables,
        operation_name="getReport",
    )

    # Optionally, you can normalize a "not found" case for convenience:
    # If call succeeded but no report exists, return a friendly hint.
    if isinstance(result, dict) and "data" in result:
        data = result["data"] or {}
        report = (data.get("basicCompliance") or {}).get("report")
        if report is None:
            return {
                "error": "NotFound",
                "detail": f"No compliance report found for symbol '{variables['symbol']}'.",
            }

    return result

## ---- tools untuk menghitung purification amount ---- ##
@tool
def get_purification_amount(purification_ratio: float, capital_gain: float = 0.0, dividend_income: float = 0.0) -> dict:
    """
    Tools untuk menghitung jumlah keuntungan yang harus disedekahkan/disumbangkan (purifikasi)
    Compute purification_amount = purification_ratio * (capital_gain + dividend_income)
    Returns:
      {
        "ratio_used": float,
        "inputs": {"capital_gain": float, "dividend_income": float},
        "amount": float
      }
    """
    try:
        r = float(purification_ratio)
        cg = float(capital_gain)
        dv = float(dividend_income)
    except Exception:
        return {"status": "error", "error": "InvalidInput", "detail": "All inputs must be numbers."}

    amount = r * (cg + dv)
    return {
        "status": "ok",
        "ratio_used": r,
        "inputs": {"capital_gain": cg, "dividend_income": dv},
        "amount": amount
    }


## ----------- Tools untuk Finnhub API ---------- ##
@tool
def get_company_overview_CP2(stock: str) -> dict:
    """

    Get company basic financian information
    
    @param stock: The stock symbol of the company
    @return: The company overview
    """

    url = f"https://finnhub.io/api/v1/stock/metric?symbol={stock}&metric=all&token=d44ckbhr01qt371ud7agd44ckbhr01qt371ud7b0"
    data = retrieve_from_endpoint(url)

    # Jika ada error (HTTP, invalid response, dsb), kembalikan apa adanya
    if "error" in data:
        return data

    # Ambil hanya bagian 'metric' (data terkini)
    latest_data = data.get("metric", {})

    return {
        "symbol": stock,
        "latest_metrics": latest_data
    }


def get_finance_agent():

    # Defined Tools
    tools = [
        get_compliance_report,
        get_purification_amount,
        get_company_overview_CP2,
    ]

    # Create the Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", f"""
                You are a Sharia-compliant finance assistant for equity screening and basic equity insights.
                Answer the following queries, being as factual and analytical as you can. 
                To obtain purification ratio, ALWAYS call get_compliance_report(symbol) first.
                If user asks to compute purification amount:
                    - Ensure both capital_gain and dividend_income are known.
                    - If missing, ask the user for the missing values in a single concise question.
                    - Then call compute_purification(purification_ratio, capital_gain, dividend_income). 
                If the user asks about financial overview/metrics/ratios or “fundamental”, call get_company_overview_CP2(stock).
                If the user says “today” or “this week/month”, translate to concrete dates relative to today.
                If you need the start and end dates but they are not explicitly provided, 
                infer from the query. Whenever you return a list of names, return also the 
                corresponding values for each name. If the volume was about a single day, 
                the start and end parameter should be the same. 
                Today's date is {datetime.today().strftime("%Y-%m-%d")}
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Initializing the LLM
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
    )

    # Create the Agent and AgentExecutor
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Add Memory to the AgentExecutor
    def get_session_history(session_id: str):

        return StreamlitChatMessageHistory(key=session_id)
    
    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_memory