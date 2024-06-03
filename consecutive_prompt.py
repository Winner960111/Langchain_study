from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0, api_key = openai_api_key)
# #prompt1
# template_question = """What is the name of the famous scientist who developed the theory of general relativity?
# Answer:"""
# prompt_question = PromptTemplate(template=template_question, input_variables=[])

# #prompt2
# template_fact = """Provide a brief description of {scientist}'s theory of general relativity.
# Answer:"""
# prompt_fact = PromptTemplate(input_variables=["scientist"], template=template_fact)

# chain_question = LLMChain(llm=llm, prompt=prompt_question)
# response_question = chain_question.run({})
# # print(response_question)
# scientist = response_question.strip()

# chain_fact = LLMChain(llm=llm, prompt= prompt_fact)

# input_data = {"scientist":scientist}

# response_fact = chain_fact.run(input_data)

# print("scientis:", scientist)
# print("Fact:",response_fact )

# Prompt 1
template_question = """What are some musical genres?
Answer: """
prompt_question = PromptTemplate(template=template_question, input_variables=[])

# Prompt 2
template_fact = """Tell me something about {genre1}, {genre2}, and {genre3} without giving any specific details.
Answer: """
prompt_fact = PromptTemplate(input_variables=["genre1", "genre2", "genre3"], template=template_fact)

# Create the LLMChain for the first prompt
chain_question = LLMChain(llm=llm, prompt=prompt_question)

# Run the LLMChain for the first prompt with an empty dictionary
response_question = chain_question.run({})
# print("this is genres:", response_question)

# Assign three hardcoded genres
genre1, genre2, genre3 = "jazz", "pop", "rock"

# Create the LLMChain for the second prompt
chain_fact = LLMChain(llm=llm, prompt=prompt_fact)

# Input data for the second prompt
input_data = {"genre1": genre1, "genre2": genre2, "genre3": genre3}

# Run the LLMChain for the second prompt
response_fact = chain_fact.run(input_data)

print("Genres:", genre1, genre2, genre3)
print("Fact:", response_fact)