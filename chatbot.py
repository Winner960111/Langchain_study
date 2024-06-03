import dotenv,os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2, api_key=api_key)

# # print(chat.invoke(
# #     [
# #         HumanMessage(
# #             content="Translate this sentence from English to French: I love programming."
# #         )
# #     ]
# # ))
# # print(chat.invoke([HumanMessage(content="What did you just say?")]))

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant. Answer all questions to the best of your ability.",
#         )
#     ]
# )

# chain = prompt | chat

# chat.invoke(
#     [
#         HumanMessage(
#             content="Translate this sentence from English to French: I love programming."
#         ),
#         AIMessage(content="J'adore la programmation."),
#         HumanMessage(content="What did you just say?"),
#     ]
# )

# demo_ephemeral_chat_history = ChatMessageHistory()

# demo_ephemeral_chat_history.add_user_message("hi!")

# demo_ephemeral_chat_history.add_ai_message("whats up?")

# # print(demo_ephemeral_chat_history.messages)

# demo_ephemeral_chat_history.add_user_message(
#     "Translate this sentence from English to French: I love programming."
# )

# response = chain.invoke({"messages": demo_ephemeral_chat_history.messages})

# demo_ephemeral_chat_history.add_ai_message(response)

# demo_ephemeral_chat_history.add_user_message("What did you just say?")

# chain.invoke({"messages": demo_ephemeral_chat_history.messages})

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, add_start_index = True)
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# k is the number of chunks to retrieve
retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k":4})

docs = retriever.invoke("how can langsmith help with testing?")

print(docs)