from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain import hub
import bs4
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=api_key)

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_ = ("post-title", "post-header", "post-content"))

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer}
)

# print("loader==>", loader.load())
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, chunk_overlap = 200, add_start_index = True
)
all_splits = text_splitter.split_documents(docs)
# print(len(all_splits))
# print(all_splits[10].metadata)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k":6})
retriever_docs = retriever.invoke("What are the approaches to Task Decomposition?")
# print(retriever_docs[0].page_content)

# use sample prompt for Q/A from hub
prompt = hub.pull("rlm/rag-prompt")
example_messages = prompt.invoke(
    {"context":"filler context", "question": "filler question"}
).to_messages()

# print(example_messages[0].content)

def format_docs(docs):
    ex= "\n\n".join(doc.page_content for doc in docs)
    print(ex)
    return ex

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} 
    | prompt
    | llm
    | StrOutputParser
)

# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Use three sentences maximum and keep the answer as concise as possible.
# Always say "thanks for asking!" at the end of the answer.

# {context}

# Question: {question}

# Helpful Answer:"""
# custom_rag_prompt = PromptTemplate.from_template(template)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | custom_rag_prompt
#     | llm
#     | StrOutputParser()
# )

print(rag_chain.invoke("What is Task Decomposition?"))