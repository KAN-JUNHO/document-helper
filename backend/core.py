import os
from typing import Any, Dict, List

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


class CustomRetriever:
    def __init__(self, vector_store, query: str, k: int):
        self.vector_store = vector_store
        self.query = query
        self.k = k
        # 유사도 점수와 함께 문서를 검색하는 메서드입니다.

    def retrieve(self) -> Dict:
        # 사전 형식의 결과를 반환하거나 기대하는 인터페이스에 따라 객체를 반환해야 합니다.
        results = self.vector_store.similarity_search_with_score(
            query=self.query, k=self.k
        )
        # 결과를 기대하는 구조에 맞게 포맷팅합니다.
        return results


def run_llm_OPENAI(
    query: str,
    search_type=None,
    chat_history: List[Dict[str, Any]] = [],
    chunk_size=None,
    chunk_overlap=None,
    search_kwargs=None,
    chain_type=None,
    selected_files=None,
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
            ),
            chain_type=chain_type,
            return_source_documents=True,
            verbose=True,
        )
    # 각 파일에 대한 결과를 responses 리스트에 추가합니다.
    responses.append(qa({"question": query, "chat_history": chat_history}))

# 모든 파일에 대한 처리가 끝난 후 responses 리스트를 반환합니다.
return responses
