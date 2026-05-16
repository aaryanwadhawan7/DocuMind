# ingest.py : takes a PDF path, creates a FIASS vector store from it.
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

VECTORSTORE_PATH = 'vectorstore/index'
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv()


def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
        model=EMBED_MODEL
    )


def ingest_pdf(doc_path: str):
    # Load Document
    loader = PyPDFLoader(file_path=doc_path)
    document = loader.load()

    # Split Document into multiple chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(document)

    # Embeddings
    embedding = get_embeddings()

    # Store embeddings in vector store
    vector_store = FAISS.from_documents(chunks, embedding)
    vector_store.save_local(VECTORSTORE_PATH)

    return len(chunks)