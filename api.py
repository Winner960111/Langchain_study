from langchain.chains.openai_functions.openapi import get_openapi_chain
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

chain = get_openapi_chain(
    "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/"
)
chain("What are some options for a men's large blue button down shirt")

# this is not working well.....