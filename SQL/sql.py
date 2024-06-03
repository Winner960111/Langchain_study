from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv
load_dotenv()

# connect of sqlite3 database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(db.dialect)
# print(db.get_usable_table_names())
# print(db.run("SELECT * FROM Artist LIMIT 10;"))

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=api_key)
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "How many employees are there"})
# print(response)
# print(db.run(response))

# print(chain.get_prompts()[0].pretty_print())

# new method
execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query
chain.invoke({"question": "How many employees are there"})

# by Prompt
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | llm |StrOutputParser()
chain = (
    RunnablePassthrough.assign(query = write_query).assign(
        result = itemgetter("query") | execute_query
    )
    | answer
)
# print(chain.invoke({"question": "How many employees are there"}))

# Advanced methods(agent)

agent_executor = create_sql_agent(llm, db = db, agent_type = "openai-tools", verbose=True)

# print(agent_executor.invoke({
#     "input": "List the total sales per country. Which country's customers spent the most?"
# }))

print(agent_executor.invoke({
"input": "Describe the playlisttrack table"
 }))