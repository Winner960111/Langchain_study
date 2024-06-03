import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
df = pd.read_csv('Titanic.csv')
# print(df.shape)
# print(df.columns.tolist())

engine = create_engine("sqlite:///titanic.db")
# df.to_sql("titanic", engine, index=False)

db = SQLDatabase(engine=engine)
# print(db.dialect)
# print(db.get_usable_table_names())
# print(db.run("SELECT * FROM titanic WHERE Age < 2;"))
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# print(agent_executor.invoke({"input":"what's the average age of survivors"}))
ai_msg = llm.invoke(
    "I have a pandas DataFrame 'df' with columns 'Age' and 'Fare'. Write code to compute the correlation between the two columns. Return Markdown for a Python code snippet and nothing else."
)
# print(ai_msg.content)

llm_with_toos = llm.bind_tools([tool], tool_choice=tool.name)
llm_with_toos.invoke(
    "I have a dataframe 'df' and want to know the correlation between the 'Age' and 'Fare' columns"
)

parser = JsonOutputKeyToolsParser(tool.name, first_tool_only=True)

(llm_with_toos | parser).invoke(
    "I have a dataframe 'df' and want to know the correlation between the 'Age' and 'Fare' columns"
)
