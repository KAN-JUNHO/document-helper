from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import (
    OpenAIEmbeddings,
)
from langchain.vectorstores.faiss import FAISS


def initialize_chat_model():
    return ChatOpenAI(verbose=True, temperature=0)


def load_vectorstore(directory, embeddings):
    return FAISS.load_local(directory, embeddings)


def run_queries(qa_model,type):
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
        print(f"질문 {type}: "+query)
        response = qa_model.run(query)
        print(f"답변 {type}: "+response)



def FAISS_load_OpenAI(chunk_size,chunk_overlap):
    print(f"FAISS_load_OpenAI chunk_size : {chunk_size}  chunk_size : {chunk_overlap}")
    chat_model = initialize_chat_model()
    embeddings = OpenAIEmbeddings()
    vectorstore = load_vectorstore(f"faiss_index_react/OpenAIEmbeddings/{chunk_size}_{chunk_overlap}", embeddings)

    for type in ["mmr", "similarity"]:
        qa_model = RetrievalQA.from_chain_type(
            llm=chat_model,
            retriever=vectorstore.as_retriever(
                search_type=type,
                search_kwargs={"k": 2, "fetch_k": 10}
            ),

        )

        run_queries(qa_model,type)


def FAISS_load_OpenAI_lambda_mult(chunk_size,chunk_overlap):
    print(f"FAISS_load_OpenAI_lambda_mult chunk_size : {chunk_size}  chunk_size : {chunk_overlap}")
    chat_model = initialize_chat_model()
    embeddings = OpenAIEmbeddings()
    vectorstore = load_vectorstore(f"faiss_index_react/OpenAIEmbeddings/{chunk_size}_{chunk_overlap}", embeddings)

    for type in ["mmr", "similarity"]:
        qa_model = RetrievalQA.from_chain_type(
            llm=chat_model,
            retriever=vectorstore.as_retriever(
                search_type=type,
                search_kwargs={'k': 2, 'lambda_mult': 0.25}
            ),

        )

        run_queries(qa_model,type)

def FAISS_load_OpenAI_search_kwargs(chunk_size,chunk_overlap):
    print(f"FAISS_load_OpenAI_search_kwargs chunk_size : {chunk_size}  chunk_size : {chunk_overlap}")
    chat_model = initialize_chat_model()
    embeddings = OpenAIEmbeddings()
    vectorstore = load_vectorstore(f"faiss_index_react/OpenAIEmbeddings/{chunk_size}_{chunk_overlap}", embeddings)

    for type in ["mmr", "similarity"]:
        qa_model = RetrievalQA.from_chain_type(
            llm=chat_model,
            retriever=vectorstore.as_retriever(
                search_type=type,
                search_kwargs={'k': 2}
            ),

        )

        run_queries(qa_model,type)


if __name__ == "__main__":
    chunk_size = [100,200,500,1000,2000]
    chunk_overlap = [0,10]

    for size in chunk_size:
        for overlap in chunk_overlap:
            if overlap == 0:
                FAISS_load_OpenAI(size,overlap)
                FAISS_load_OpenAI_lambda_mult(size,overlap)
                FAISS_load_OpenAI_search_kwargs(size,overlap)
            else:
                FAISS_load_OpenAI(size, int(size/overlap))
                FAISS_load_OpenAI_lambda_mult(size, int(size/overlap))
                FAISS_load_OpenAI_search_kwargs(size, int(size/overlap))