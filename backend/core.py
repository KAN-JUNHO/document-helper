import os
from typing import Any, Dict, List

from langchain.embeddings import (
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from langchain.vectorstores import FAISS
import pinecone

from consts import INDEX_NAME

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        verbose=True,
    )

    return qa({"question": query, "chat_history": chat_history})


def run_llm_OPENAI(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )
    new_vectorestore = FAISS.load_local(
        "faiss_index_react/HuggingFaceEmbeddings", embeddings
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=new_vectorestore.as_retriever(),
        return_source_documents=True,
        verbose=True,
    )

    return qa({"question": query, "chat_history": chat_history})


def run_llm_hugging(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = HuggingFaceEmbeddings(
        api_key="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",
        model_name="sentence-transformers/all-MiniLM-l6-v2",
    )

    new_vectorestore = FAISS.load_local(
        "faiss_index_react/HuggingFaceEmbeddings", embeddings
    )

    qa = ConversationalRetrievalChain.from_llm(
        retriever=new_vectorestore.as_retriever(),
        return_source_documents=True,
        verbose=True,
    )

    return qa({"question": query, "chat_history": chat_history})
