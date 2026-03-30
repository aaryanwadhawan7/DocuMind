# main.py : FastAPI app with two endpoints: /upload and /ask


import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.ingest import ingest_pdf
from app.query import answer_question
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="DocuMind", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploads"

Instrumentator().instrument(app).expose(app)
# Instrumentator : creates a monitoring employee
# instrument(app) : tells to monitor the FastAPI app
# expose(app) : creates a /metrics endpoint for Prometheus to visit every 15 secs for analytics

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # Save uploaded file to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run the ingest pipeline
    num_chunks = ingest_pdf(file_path)

    return {
        "message": f"Indexed {file.filename} successfully",
        "chunks_created": num_chunks
    }

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = answer_question(request.question)
    return result
