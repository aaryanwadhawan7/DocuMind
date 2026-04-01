# query.py - Answer a question using the saved vectorstore + Groq LLM

import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import mlflow

from dotenv import load_dotenv
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

load_dotenv() # loads GROQ_API_KEY from .env file

VECTORSTORE_PATH = 'vectorstore/index' #path where all the vectors will be stored
#these variables can be easily changed if someone wants to find out which model is best/efficient
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'Llama-3.1-8B-Instant'

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI','http://mlflow:5000')

# fixing bug 
# print (f"MLFLOW_URI: {MLFLOW_URI}")

if MLFLOW_URI:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("documind-queries")

def answer_question(question: str) -> dict:

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # required by LangChain for local files
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # temperature = 0 means consistent answers
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=LLM_MODEL, 
        temperature=0
    )

    """
    RetrievalQA chain = retriever + LLM wired together
    chain_type="stuff" means: stuff all retrieved chunks into one prompt
    return_source_documents=True means: also return which chunks were used
    """
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents = True,
        chain_type="stuff"
    )
    
    # This ensures result and latency always exist
    start_time = time.time()
    result     = chain.invoke({"query": question})
    latency    = round(time.time() - start_time, 3) 

    if MLFLOW_URI:
        try:
            with mlflow.start_run():
                # log_param = settings used (things that don't change mid-run)
                mlflow.log_param("embed_model", EMBED_MODEL)
                mlflow.log_param("llm_model",   LLM_MODEL)
                mlflow.log_param("chunk_k",     3)
                mlflow.log_param("temperature", 0)
                mlflow.log_param("question",    question)
                mlflow.log_metric("latency_seconds",     latency)
                mlflow.log_metric("source_chunks_used",  len(result["source_documents"]))
        except Exception:
            #If mlflow logging fails for some reason, user will still get the answers from llm
            pass

    sources = sorted(set([
        doc.metadata.get("page", 0) + 1
        for doc in result["source_documents"]
    ]))

    return {
        "answer":           result["result"],
        "source_pages":     sources,
        "latency_seconds":  latency       
    }