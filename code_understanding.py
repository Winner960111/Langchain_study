import os
from dotenv import load_dotenv
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Define the repository path on your local machine
repo_path = "C:/Users/APOLLO/Desktop/test_repo"

# URL of the remote repository
repo_url = "https://github.com/langchain-ai/langchain"

# Cloning the repository
Repo.clone_from(repo_url, to_path=repo_path)

# # Load
# loader = GenericLoader.from_filesystem(
#     repo_path + "/libs/core/langchain_core",
#     glob="**/*",
#     suffixes=[".py"],
#     exclude=["**/non-utf8-encoding.py"],
#     parser = LanguageParser(language=Language.PYTHON, parser_threshold=500),
# )
# documents = loader.load()
# print(len(documents))