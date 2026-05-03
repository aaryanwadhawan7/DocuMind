# ingest.py : takes a PDF path, creates a FIASS vector store from it.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTORSTORE_PATH = 'vectorstore/index'

def get_embeddings():
    # Use a tiny model that fits in 512MB
    # all-MiniLM-L6-v2 is ~90MB on disk, ~200MB in RAM — fits fine
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def ingest_pdf (doc_path: str):
    # Load Document
    loader = PyPDFLoader(file_path = doc_path)
    document = loader.load()

    # Split Document into multiple chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 150
    )

    chunks = text_splitter.split_documents(document)

    # Embedddings
    embedding = get_embeddings()

    # Store embeddings in vector store
    vector_store = FAISS.from_documents(chunks, embedding)
    vector_store.save_local(VECTORSTORE_PATH)

    return len(chunks)

    


