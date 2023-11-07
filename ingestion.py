import os

import pinecone
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.embeddings import (
    OpenAIEmbeddings,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Pinecone 초기화
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def load_and_process_docs(path, encodings, chunk_size, chunk_overlap, separators):
    loader = UnstructuredWordDocumentLoader(path, encodings=encodings)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
    )
    documents = text_splitter.split_documents(raw_documents)

    # for doc in documents:
    #     new_url = doc.metadata["source"].replace("langchain-docs", "https:/")
    #     doc.metadata.update({"source": new_url})

    return documents


def ingest_docs_korea(size, overlap, document_name):
    chunk_size = size
    if overlap == 0:
        chunk_overlap = overlap
    else:
        chunk_overlap = int(size / overlap)
    documents = load_and_process_docs(
        path="langchain-docs/langchain.readthedocs.io/en/latest/fairy_tails/classic_fairy_tales.docx",
        encodings="utf-8",
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["."],
    )

    embeddings_list = [OpenAIEmbeddings()]

    for embedding in embeddings_list:
        vectorstore = FAISS.from_documents(documents, embedding)
        embedding_name = embedding.__class__.__name__
        if hasattr(embedding, "model_name"):
            vectorstore.save_local(
                f"faiss_index_react/{document_name}/{embedding.model_name}/{chunk_size}_{chunk_overlap}"
            )
        else:
            vectorstore.save_local(
                f"faiss_index_react/{document_name}/{chunk_size}_{chunk_overlap}"
            )
        print(f"Loading to vectorestore/{document_name}/{embedding_name}/ done")


def chunk_make(document_name):
    chunk_size = [100, 200, 500, 1000, 1800]
    chunk_overlap = [0, 10]
    for size in chunk_size:
        for overlap in chunk_overlap:
            ingest_docs_korea(size, overlap, document_name)


if __name__ == "__main__":
    # ingest_docs_english()
    chunk_make()

    # ingest_docs_korea()
