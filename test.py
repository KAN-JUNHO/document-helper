from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS


def initialize_chat_model():
    return ChatOpenAI(verbose=True, temperature=0)


def load_vectorstore(directory, embeddings):
    return FAISS.load_local(directory, embeddings)


def run_queries(qa_model):
    queries = ["헨젤과 그레텔 이야기해줘", "헨젤과 그레텔에서 고양이가 나와?", "고양이 나오는 동화 전부다 이야기 해줘"]

    for query in queries:
        response = qa_model.run(query)
        print(response)


def FAISS_load_HuggingFace():
    print("hugging")

    chat_model = initialize_chat_model()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = load_vectorstore(
        "faiss_index_react/HuggingFaceEmbeddings", embeddings
    )
    qa_model = RetrievalQA.from_chain_type(
        llm=chat_model, retriever=vectorstore.as_retriever()
    )

    run_queries(qa_model)


def FAISS_load_OpenAI():
    print("openai")
    chat_model = initialize_chat_model()
    embeddings = OpenAIEmbeddings()
    vectorstore = load_vectorstore("faiss_index_react/OpenAIEmbeddings", embeddings)
    qa_model = RetrievalQA.from_chain_type(
        llm=chat_model, retriever=vectorstore.as_retriever()
    )

    run_queries(qa_model)


if __name__ == "__main__":
    FAISS_load_HuggingFace()
    FAISS_load_OpenAI()
