from langchain_openai import ChatOpenAI
from dotenv import dotenv_values
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import ChatMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool
from src.tools.pandasTool import tool
import os

def get_agent():
    
    tools = [tool]
    tool_names = [tool.name for tool in tools ]
    tools_description = "\n".join([tool.name + ": " + tool.description for tool in tools])

    prompt_str = f"""
    You are friendly HR assistant. You are tasked to assist the current user on questions related to HR. You have access to the following tools:
    {tools_description}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of {tools}
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: [input]
    Thought:[agent_scratchpad]
    
    """

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template="Your a helpful assistant")),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["input"], template='{input}')),
        MessagesPlaceholder(variable_name='agent_scratchpad')])


    agent = create_openai_tools_agent(tools=tools, llm=llm, prompt=chat_template)
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    runnable_agent = RunnableWithMessageHistory(
        AgentExecutor(agent=agent, tools=tools, verbose=True),
        get_session_history, input_messages_key="input", history_messages_key="chat_history")

    return runnable_agent


# op = agent_with.invoke({'input':"How many number of employees are available?"},
#                  config={"configurable":{"session_id":"1234"},
#                          })
# print(op)
# print(store)