import os
from typing import Any, Dict, List

import pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import faiss

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def run_llm_OPENAI(
        query: str,
        search_type=None,
        chat_history: List[Dict[str, Any]] = [],
        chunk_size=None,
        chunk_overlap=None,
        search_kwargs=None,
        chain_type=None,
        selected_files=None,
        similarity_search_with_score=None
):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )
    # 모든 selected_files에 대한 결과를 저장할 리스트
    responses = []

    for selected_file in selected_files:
        # new_vectorestore = FAISS.load_local(f"faiss_index_react/fairy_tales/{chunk_size}_{chunk_overlap}", embeddings)

        vectorestore_path = (
            f"faiss_index_react/{selected_file}/{chunk_size}_{chunk_overlap}"
        )
        new_vectorestore = FAISS.load_local(vectorestore_path, embeddings)



        qa = ConversationalRetrievalChain.from_llm(
            llm=chat,
            retriever=new_vectorestore.as_retriever(
                search_type=search_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                search_kwargs=search_kwargs,
                similarity_search_with_score=similarity_search_with_score
            ),
            chain_type=chain_type,
            return_source_documents=True,
            verbose=True,
        )
        # 각 파일에 대한 결과를 responses 리스트에 추가합니다.
        responses.append(qa({"question": query, "chat_history": chat_history,
                             "similarity_search_with_score": new_vectorestore.similarity_search_with_score(query)}))

    # 모든 파일에 대한 처리가 끝난 후 responses 리스트를 반환합니다.
    return responses
