#example 1

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
load_dotenv()
# # Define your desired data structure.
# class Suggestions(BaseModel):
#     words: List[str] = Field(description="list of substitue words based on context")
#     reasons: List[str] = Field(description="the reasoning of why this word fits the context")

# parser = PydanticOutputParser(pydantic_object=Suggestions)

# missformatted_output = '{"words":["conduct", "manner"], "reasoning":["refers to the way someone acts in a particular situation.", "refers to the way someone behaves in a particular situation."]}'

# print(parser.invoke(missformatted_output))

#example 2

from langchain.llms import OpenAI
from langchain.output_parsers import OutputFixingParser
import os

api_key = os.getenv("OPENAI_API_KEY")
model = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0.0, api_key = api_key)

# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")

parser = PydanticOutputParser(pydantic_object=Suggestions)

missformatted_output = '{"words": ["conduct", "manner"]}'

outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
print(outputfixing_parser.parse(missformatted_output))