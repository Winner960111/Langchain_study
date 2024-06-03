from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
# Initialize LLM

llm = ChatOpenAI(model_name = "gpt-3.5-turbo-0125", temperature=0, api_key = openai_api_key)

template = """As a futuristic robot band conductor, I need you to help me come up with a song title. What's a cool song title for a song about {theme} in the year {year}"""

prompt = PromptTemplate(
    input_variables = ["theme", "year"],
    template = template
)

llm = ChatOpenAI(model_name = "gpt-3.5-turbo-0125", temperature=0, api_key = openai_api_key)

input_data = {"theme":"interstellar travel", "year":"3030"}

chain = LLMChain(llm=llm, prompt = prompt)

response = chain.run(input_data)

print("Theme:interstellar travel")
print("Year:3030")
print("AI-generated song title:", response)
