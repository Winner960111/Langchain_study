from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser, StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY") 

response_schemas = [
    ResponseSchema(name="words", description="A substitue word based on context"),
    ResponseSchema(name="reasons", description="the reasoning of why this word fits the context.")
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)
# Prepare the Prompt
template = """
Offer a list of suggestions to substitute the specified target_word based on the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format(
  target_word="behaviour",
  context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

# Loading OpenAI API
model = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0.0, api_key = api_key)

# Send the Request
output = model(model_input)
print(parser.parse(output))