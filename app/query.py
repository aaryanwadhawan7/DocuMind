# query.py - Answer a question using the saved vectorstore + Groq LLM

import os
import time
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

import mlflow

load_dotenv()

VECTORSTORE_PATH = "vectorstore/index"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.1-8b-instant"

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")


def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
        model=EMBED_MODEL
    )


def setup_mlflow():
    if MLFLOW_URI:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("documind-queries")
        except Exception:
            pass


llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0
)


prompt_template = """
You are an intelligent assistant.

Use ONLY the provided context to answer the question.

Make your answer:
- Detailed
- Structured (use bullet points if helpful)
- Clear and well-explained

If the answer is not in the context, say:
"I could not find this in the document."

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


def answer_question(question: str) -> dict:

    setup_mlflow()

    # Load embeddings
    embeddings = get_embeddings()

    # Check vectorstore exists
    if not os.path.exists(f"{VECTORSTORE_PATH}/index.faiss"):
        return {
            "answer": "No document has been uploaded yet. Please upload a PDF first.",
            "source_pages": [],
            "latency_seconds": 0
        }

    # Load vectorstore
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # Create RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Measure latency
    start_time = time.time()

    result = chain.invoke({
        "query": question
    })

    latency = round(time.time() - start_time, 3)

    # MLflow logging
    if MLFLOW_URI:
        try:
            with mlflow.start_run():

                mlflow.log_param("embed_model", EMBED_MODEL)
                mlflow.log_param("llm_model", LLM_MODEL)
                mlflow.log_param("chunk_k", 3)
                mlflow.log_param("temperature", 0)
                mlflow.log_param("question", question)

                mlflow.log_metric("latency_seconds", latency)

                mlflow.log_metric(
                    "source_chunks_used",
                    len(result["source_documents"])
                )

        except Exception:
            pass

    # Extract source pages
    sources = sorted(set([
        doc.metadata.get("page", 0) + 1
        for doc in result["source_documents"]
    ]))

    return {
        "answer": result["result"],
        "source_pages": sources,
        "latency_seconds": latency
    }