from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List
import json
import os
from dotenv import load_dotenv
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
# Define your desired data structure.

# simple output

# class Suggestions(BaseModel):
#     words: List[str] = Field(description="list of substitue words based on context")

#     # Throw error in case of receiving a numbered-list from API
#     @validator('words')
#     def not_start_with_number(cls, field):
#         for item in field:
#             if item[0].isnumeric():
#                 raise ValueError("The word can not start with numbers!")
#         return field
# parser = PydanticOutputParser(pydantic_object=Suggestions)

# template = """
# Offer a list of suggestions to substitue the specified target_word based the presented context.
# {format_instructions}
# target_word = {target_word}
# context = {context}
# """

# multiple output
template = """
Offer a list of suggestions to substitute the specified target_word based on the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""

class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")
    @validator('words')
    def not_start_wtith_number(cls, field):
        for item in field:
            if item[0].isnumeric():
                raise ValueError("The word can not start with numbers!")
        return field

    @validator("reasons")
    def end_with_dot(cls, field):
        for idx, item in enumerate(field):
            if item[-1] != ".":
                field[idx] += "."
        return field

parser = PydanticOutputParser(pydantic_object=Suggestions)

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format_prompt(
    target_word="behaviour",
    context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

# Before executing the following code, make sure to have your OpenAI key saved in the "OPENAI_API_KEY" environment variable.

model = ChatOpenAI(model_name='gpt-3.5-turbo-0125', temperature=0, api_key=openai_key)

output = model(model_input.to_messages())
print(parser.invoke(output))