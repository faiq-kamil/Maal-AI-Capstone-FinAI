import streamlit as st
from tools_capstone import get_finance_agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from datetime import datetime


### ----- Page Configuration ----- ###
st.set_page_config(
    page_title="Financial Chatbot",
    page_icon="💰",
    layout="centered"
)

st.header("Ask Maal AI - your sharia investing advisor💸", divider='green')

### ------ managing chat session
if "selectbox_selection" not in st.session_state:
    st.session_state['selectbox_selection'] = ["Default Chat"]

selectbox_selection = st.session_state['selectbox_selection']

if st.sidebar.button("✍️ Create New Chat", use_container_width=True):
    selectbox_selection.append(f"New Chat - {datetime.now().strftime('%H:%M:%S')}")

session_id = st.sidebar.selectbox("Chats", options=selectbox_selection, index=len(selectbox_selection)-1)

st.sidebar.image("Logo Maal AI.png")
## ---------- About
st.sidebar.markdown("#")
st.sidebar.markdown('''
                    
ℹ️ **Maal AI**

Financial Advisor Chatbot to help you achieve wealth through sharia-compliance investing.
                    
🛠️ **I can help you with:**
                    
**- 🏢 Stock Sharia-Compliant Report**              
**- 🧮 Purification calculator**                                           
**- 🔍 Stock basic financial information**
               
''')

## ------- displaying chat history
chat_history = StreamlitChatMessageHistory(key=session_id)

for message in chat_history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)


## ------- Chat Interface
prompt=st.chat_input("how can I help you?")
agent = get_finance_agent()

if prompt:
    with st.chat_message("human"):
        st.markdown(prompt)

    with st.chat_message("ai"):
        response = agent.invoke({"input": prompt}, config={"configurable": {"session_id": session_id}})
        st.markdown(response['output'])


