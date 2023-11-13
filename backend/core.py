import os
from typing import Any, Dict
from typing import List

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import BaseRetriever
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer



def run_llm_OPENAI(
    query: str,
    search_type=None,
    chat_history: List[Dict[str, Any]] = [],
    chunk_size=None,
    chunk_overlap=None,
    search_kwargs=None,
    chain_type=None,
    selected_files=None,
    selected_chat_model=None
):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    if selected_chat_model == "gpt-3.5-turbo":
        chat = ChatOpenAI(
            verbose=True,
            temperature=0,
        )
    elif selected_chat_model == "kfkas/Llama-2-ko-7b-Chat":
        # tokenizer = AutoTokenizer.from_pretrained("kfkas/Llama-2-ko-7b-Chat")
        chat = AutoModelForCausalLM.from_pretrained("kfkas/Llama-2-ko-7b-Chat")

    # 모든 selected_files에 대한 결과를 저장할 리스트
    responses = []

    for selected_file in selected_files:
        # new_vectorestore = FAISS.load_local(f"faiss_index_react/fairy_tales/{chunk_size}_{chunk_overlap}", embeddings)

        vectorestore_path = (
            f"faiss_index_react/{selected_file}/text-embedding-ada-002/{chunk_size}_{chunk_overlap}"
        )
        new_vectorestore = FAISS.load_local(vectorestore_path, embeddings)

        class MyConcreteRetriever(BaseRetriever):
            search_type: str = Field(...)
            search_kwargs: Dict[str, Any] = Field(default_factory=dict)
            tags: List[str] = Field(default_factory=list)  # tags 필드를 추가합니다.
            vectorstore: FAISS = Field(...)

            def __init__(
                self,
                search_type: str,
                vectorstore: FAISS,
                search_kwargs: Dict[str, Any],
                tags: List[str],
                *args,
                **kwargs,
            ):
                super().__init__(*args, **kwargs)
                self.search_type = search_type
                self.vectorstore = vectorstore
                self.search_kwargs = search_kwargs
                self.tags = tags

            def _get_relevant_documents(self, query):
                if "similarity" == self.search_type:
                    results_with_scores = new_vectorestore.similarity_search_with_score(
                        query=query,
                        k=self.search_kwargs["k"],
                        fetch_k=self.search_kwargs["fetch_k"],
                    )

                    documents_with_scores = [
                        Document(
                            page_content=document.page_content,
                            metadata={**document.metadata, "score": 1 - score},
                        )
                        for document, score in results_with_scores
                    ]
                    # 결과를 ConversationalRetrievalChain이 처리할 수 있는 형식으로 변환
                elif "similarity_score_threshold" == self.search_type:
                    results_with_scores = (
                        new_vectorestore.similarity_search_with_relevance_scores(
                            query=query,
                            k=self.search_kwargs["k"],
                            score_threshold=self.search_kwargs["score_threshold"],
                        )
                    )

                    documents_with_scores = [
                        Document(
                            page_content=document.page_content,
                            metadata={**document.metadata, "score": score},
                        )
                        for document, score in results_with_scores
                    ]
                elif "mmr" == self.search_type:
                    results_with_scores = new_vectorestore.max_marginal_relevance_search_with_score_by_vector(
                        embedding=embeddings.embed_query(query),
                        k=self.search_kwargs["k"],
                        lambda_mult=self.search_kwargs["lambda_mult"],
                        fetch_k=self.search_kwargs["fetch_k"],
                    )

                    documents_with_scores = [
                        Document(
                            page_content=document.page_content,
                            metadata={**document.metadata, "score": 1 - score},
                        )
                        for document, score in results_with_scores
                    ]

                return documents_with_scores

        my_retriever_instance = MyConcreteRetriever(
            search_type=search_type,
            vectorstore=new_vectorestore,
            search_kwargs=search_kwargs,
            tags=["FAISS", "OpenAIEmbeddings"],  # 태그 설정
        )

        # ConversationalRetrievalChain 생성
        qa = ConversationalRetrievalChain.from_llm(
            llm=chat,
            retriever=my_retriever_instance,
            chain_type=chain_type,
            return_source_documents=True,
            verbose=True,
        )
        # qa = ConversationalRetrievalChain.from_llm(
        #     llm=chat,
        #     retriever=new_vectorestore.as_retriever(
        #         search_type=search_type,
        #         chunk_size=chunk_size,
        #         chunk_overlap=chunk_overlap,
        #         search_kwargs=search_kwargs,
        #     ),
        #     chain_type=chain_type,
        #     return_source_documents=True,
        #     verbose=True,
        # )

        # 각 파일에 대한 결과를 responses 리스트에 추가합니다.
        responses.append(qa({"question": query, "chat_history": chat_history}))

    # 모든 파일에 대한 처리가 끝난 후 responses 리스트를 반환합니다.
    return responses
