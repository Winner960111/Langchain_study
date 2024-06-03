from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

with open(".\script.txt", "r") as file:
    input_text = file.read()
llm = ChatOpenAI(model_name = "gpt-3.5-turbo-0125", temperature=0, api_key = openai_api_key)

examples = [
    {"question": "What is your name?", "answer": "My name is HaoMing."},
    {"question":"What language do you prefer?", "answer": "English. That is why I am an singaporean."},
    {"question":"Where do you live?", "answer": "I live in Singapore."},
]

example_formatter_template = """
question:{question}
answer:{answer}\n
"""

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template=example_formatter_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Here are some examples of questions and the answers associated with them:\n\n",
    suffix="\n\nNow, extract all of questions and its answers from {input}. You should output as json format.",
    input_variables=["input"],
    # example_selector="\n"
)

formatted_prompt = few_shot_prompt.format(input=input_text)

chain = LLMChain(llm=llm, prompt=PromptTemplate(template=formatted_prompt, input_variables=[]))
response = chain.run({})

print(response)