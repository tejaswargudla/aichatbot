import streamlit as st
import langfuse
import os
from dotenv import dotenv_values
from langfuse.callback import CallbackHandler
cfg = dotenv_values(".env")
 
# get keys for your project from https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = cfg["LANGFUSE_PUBLIC_KEY"]
os.environ["LANGFUSE_SECRET_KEY"] = cfg["LANGFUSE_SECRET_KEY"]
os.environ["LANGFUSE_HOST"] = cfg["LANGFUSE_HOST"]
from src.agents.agent import get_agent
langfuse_handler = CallbackHandler()


st.title("Echo Bot")
prompt = "Hello"
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    agent = get_agent()
    op = agent.invoke({'input':prompt},
                     config={"configurable":{"session_id":"12345"},
                             'callbacks':[langfuse_handler]
                             })
    
    response = f"Echo: {op.get('output','Nope')}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})