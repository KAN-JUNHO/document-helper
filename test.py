import pandas as pd

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import (
    OpenAIEmbeddings,
)
from langchain.vectorstores.faiss import FAISS

size = None  # 초기화
overlap = None  # 초기화

def initialize_chat_model():
    return ChatOpenAI(verbose=True, temperature=0)


def load_vectorstore(directory, embeddings):
    return FAISS.load_local(directory, embeddings)


def run_queries(qa_model,search_type,chain_type,search_kwargs):
    queries = [
        "헨젤과 그레텔 이야기를 요약해줘"
        # "헨젤과 그레텔에서 마녀가 나와?",
        # "헨젤과 그레텔에서 고양이의 정체가 머였어?",
        # "마녀는 무엇으로 헨젤과 그레텔을 유인했어?",
        # "라푼첼 동화에서 왕자님이 나와?",
        # "라푼첼은 성에서 무엇을 이용해서 탈출했어?"
        # "고양이가 나오는 동화를 말해줘",
        # "우주와 관련된 동화가 있어?",
        # "헨젤과 그레텔은 길을 잃버러지 않게 무엇을 하니?",
        # "라푼첼에서 왕자는 가시에 눈을 찔려?"
    ]

    for query in queries:
        # print(f"질문 {type}: "+query)
        response = qa_model.run(query)
        print(f"답변 {search_type}_{chain_type}: "+response)
        results.append({"size" : f"{size}","overlap" : f"{overlap}","search_type" : f"{search_type}","chain_type" : f"{chain_type}", "search_kwargs" : f"{search_kwargs}", "질문": query, "답변": response})

def process_with_search_kwargs(chat_model, vectorstore, search_kwargs):
    print(f"search_kwargs={search_kwargs}")
    for search_type in ["mmr", "similarity"]:
        for chain_type in ["stuff", "map_reduce", "refine"]:
            qa_model = RetrievalQA.from_chain_type(
                llm=chat_model,
                retriever=vectorstore.as_retriever(
                    search_type=search_type,
                    search_kwargs=search_kwargs
                ),
                chain_type=chain_type
            )
            run_queries(qa_model, search_type, chain_type,search_kwargs)

if __name__ == "__main__":
    chunk_size = [100,200,500,1000,1800]
    chunk_overlap = [0,10]
    # chunk_size=[100]
    # chunk_overlap=[0]
    results = []
    for size in chunk_size:
        for overlap in chunk_overlap:

            chat_model = initialize_chat_model()
            embeddings = OpenAIEmbeddings()
            if overlap == 0:
                vectorstore = load_vectorstore(f"faiss_index_react/OpenAIEmbeddings/{size}_{overlap}", embeddings)
                print(f"chunk_size={size}, chunk_overlap={overlap}")
            else:
                vectorstore = load_vectorstore(f"faiss_index_react/OpenAIEmbeddings/{size}_{int(size/overlap)}", embeddings)
                print(f"chunk_size={size}, chunk_overlap={int(size/overlap)}")

            process_with_search_kwargs(chat_model, vectorstore, {"k": 2, "fetch_k": 10})
            process_with_search_kwargs(chat_model, vectorstore, {'k': 2, 'lambda_mult': 0.25})
            process_with_search_kwargs(chat_model, vectorstore, {'k': 2})

            print("---------------------------------------------------------------")

    # 결과를 DataFrame으로 변환하고 엑셀 파일로 저장
    df = pd.DataFrame(results)
    df.to_excel("C:\\Users\\Admin\\PycharmProjects\\document-helper\\faiss_index_react\\results.xlsx", index=False, engine='openpyxl')