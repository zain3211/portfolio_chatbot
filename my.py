import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings
)
from langchain.storage import InMemoryStore
from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import PromptTemplate
openai_api_key=os.getenv("OPENAI_API_KEY")

loaders = [
    TextLoader("aboutwissam.txt"),
    TextLoader("albahar.txt"),
    TextLoader("bachelorthesis.txt"),
    TextLoader("gitec.txt"),
    TextLoader("languages.txt"),
    TextLoader("masterthesis.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#Load the embedings
vector_db = Chroma(persist_directory='vector_db',collection_name="wissam_collection",embedding_function=embeddings)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
vectorstore = Chroma(
    collection_name="wissam_cv", embedding_function=embeddings
)
#Memory Store
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)
retriever.add_documents(docs, ids=None)

#Prompt Template
template = """
You are a helpful assistant that can answer questions about Wissam.answer the questions of user in a helpful way.only give answer from context if question is out of context just say i dont know.Finally, if you don't know the answer about wissam, simply state that you don't know the answer and that Wissam can be contacted through e-mail present on his CV.
Question: {question}
Context: {context}
Assistant Response:
"""
propmpt=PromptTemplate.from_template(template=template)
llm=ChatOpenAI(api_key=openai_api_key)

memory=ConversationBufferMemory(return_messages=True,memory_key="chat_history")
chain=ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": propmpt},
    chain_type="stuff"
)

def my_main(user_input:str) -> str:
    """
    This function takes the user question or query and returns the response
    :param: user_input: The input text from the user
    :return: String value of answer  to the user question or query
    """
    response=chain.invoke({"question":user_input,"chat_history": st.session_state.messages})
    return response.get("answer")


