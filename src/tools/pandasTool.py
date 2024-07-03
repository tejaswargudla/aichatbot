from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values
from langchain.agents import AgentType, Tool
import pandas as pd
import os
import sys

path = os.getcwd()
if "/aichatbot" in path:
    path = path.rsplit("/aichatbot", 1)[0]
    path = path + "/aichatbot"
    if path not in sys.path:
        sys.path.insert(0, path)
    cfg = dotenv_values(f'{path}/.env')
else:
    cfg = dotenv_values(".env")

os.environ["OPENAI_API_KEY"] = cfg["OPENAI_KEY"]

df = pd.read_csv(f"{path}/src/data/employee_data.csv")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
pandas_agent = create_pandas_dataframe_agent(llm,
                                             df,
                                             verbose=True,
                                             agent_type=AgentType.OPENAI_FUNCTIONS,
                                             number_of_head_rows=1,
                                             allow_dangerous_code=True)

tool = Tool(
    name="EmployeeData",
    description="Useful for when you need to answer questions about employee data stored in pandas dataframe 'df'.",
    func=pandas_agent.run,
    )