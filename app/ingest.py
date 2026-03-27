# ingest.py : takes a PDF path, creates a FIASS vector store from it.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTORSTORE_PATH = 'vectorstore/index'

def ingest_pdf (doc_path: str):
    # Load Document
    loader = PyPDFLoader(file_path = doc_path)
    document = loader.load()

    # Split Document into multiple chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    chunks = text_splitter.split_documents(document)

    # Embedddings
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store embeddings in vector store
    vector_store = FAISS.from_documents(chunks, embedding)
    vector_store.save_local(VECTORSTORE_PATH)

    


