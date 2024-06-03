import os
from dotenv import load_dotenv
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

