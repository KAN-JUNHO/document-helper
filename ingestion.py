import os
import pinecone
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Pinecone
from consts import INDEX_NAME

# Pinecone 초기화
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def load_and_process_docs(
    path, encodings="utf-8", chunk_size=1000, chunk_overlap=500, separators=None
):
    loader = UnstructuredWordDocumentLoader(path, encodings=encodings)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
    )
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"].replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    return documents


def ingest_docs_korea():
    documents = load_and_process_docs(
        "langchain-docs/langchain.readthedocs.io/en/latest/fairy_tails/fairy_tales.docx",
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", "."],
    )



    embeddings_list = [
        # OpenAIEmbeddings(),

        HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
        HuggingFaceEmbeddings(model_name="google/canine-s"),
        HuggingFaceEmbeddings(model_name="gogamza/kobart-base-v2"),
        HuggingFaceEmbeddings(model_name="nielsr/lilt-xlm-roberta-base"),
        HuggingFaceEmbeddings(model_name="Blaxzter/LaBSE-sentence-embeddings"),

    ]

    for embedding in embeddings_list:
        vectorstore = FAISS.from_documents(documents, embedding)
        embedding_name = embedding.__class__.__name__
        vectorstore.save_local(f"faiss_index_react/{embedding_name}_{embedding.model_name}")
        print(f"Loading to vectorestore {embedding_name} done")


if __name__ == "__main__":
    # ingest_docs_english()
    ingest_docs_korea()
