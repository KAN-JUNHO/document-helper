import os

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import ReadTheDocsLoader, Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter,TextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pinecone

from consts import INDEX_NAME

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def ingest_docs():
    # loader = ReadTheDocsLoader(path="langchain-docs/langchain.readthedocs.io/en/latest", encoding="utf-8")
    loader2 = UnstructuredWordDocumentLoader(
        "langchain-docs/langchain.readthedocs.io/en/latest/fairy_tails/classic_fairy_tales_english.docx", encodings='utf-8')

    # raw_documents1 = loader.load()
    raw_documents2 = loader2.load()

    # print(f"loaded {len(raw_documents1)} documents")
    print(f"loaded {len(raw_documents2)} documents")

    raw_documents = raw_documents2
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=500, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    embeddings = OpenAIEmbeddings()
    print(f"Going to add {len(documents)} to Pinecone")
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorestore done ***")


def ingest_docs_korea():
    # loader = ReadTheDocsLoader(path="langchain-docs/langchain.readthedocs.io/en/latest", encoding="utf-8")
    loader2 = UnstructuredWordDocumentLoader("langchain-docs/langchain.readthedocs.io/en/latest/fairy_tails/fairy_tales.docx")

    # raw_documents1 = loader.load()
    raw_documents2 = loader2.load()

    # print(f"loaded {len(raw_documents1)} documents")
    print(f"loaded {len(raw_documents2)} documents")

    raw_documents = raw_documents2
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )

    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    embeddings = OpenAIEmbeddings()
    print(f"Going to add {len(documents)} to Pinecone")
     # Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index_react")
    print("****Loading to vectorestore done ***")


def ingest_docs_english():
    # loader = ReadTheDocsLoader(path="langchain-docs/langchain.readthedocs.io/en/latest", encoding="utf-8")
    loader2 = UnstructuredWordDocumentLoader(
        "langchain-docs/langchain.readthedocs.io/en/latest/fairy_tails/classic_fairy_tales_english.docx")

    # raw_documents1 = loader.load()
    raw_documents2 = loader2.load()

    # print(f"loaded {len(raw_documents1)} documents")
    print(f"loaded {len(raw_documents2)} documents")

    raw_documents = raw_documents2
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    embeddings = OpenAIEmbeddings()

    print(f"Going to add {len(documents)} to Pinecone")
    # Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index_react")
    print("****Loading to vectorestore done ***")


def ingest_docs_query():
    english_question1 = "What did the mountain spirit turn into?"
    korean_question1 = "산신령은 무엇으로 변신했는가?"
    english_question2 = "토끼와 거북이 중에 누가 이겼는가?"
    korean_question2 = "What did the mountain spirit turn into?"

    # 질문들 리스트 생성
    questions = [english_question1, korean_question1, english_question2, korean_question2]

    # Pinecone 인덱스 초기화
    index = pinecone.Index("langchain-doc-index")

    # 각 질문을 임베딩하고 고유한 ID를 생성
    embeddings = OpenAIEmbeddings()  # OpenAI 임베딩 모델 인스턴스화 (이 부분은 실제 사용하는 임베딩 모델에 맞춰 수정해야 함)

    # Pinecone.from_texts(questions, embeddings, index_name=INDEX_NAME)
    vectorstore = FAISS.from_documents(questions, embeddings)
    vectorstore.save_local("faiss_index_react")
    print("****Loading to vectorestore done ***")

def FAISS_load():
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings()
    new_vectorestore = FAISS.load_local("faiss_index_react", embeddings)
    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="map_rerank", retriever=new_vectorestore.as_retriever())

    # print(new_vectorestore['index_to_docsotre_id'])

    res = qa.run("헨젤과 그레텔 이야기해줘")
    res2 = qa.run("헨젤과 그레텔에서 고양이가 나와?")
    res3 = qa.run("고양이 나오는 동화 전부다 이야기 해줘")

    print(res)
    print(res2)
    print(res3)
if __name__ == "__main__":
    # ingest_docs()
    # ingest_docs_korea()
    # ingest_docs_english()
    # ingest_docs_query()
    FAISS_load()