# DocuMind — AI Document Q&A

Upload a PDF and ask questions in plain English. Answers are grounded in the document and return the source pages.

**GitHub:** https://github.com/aaryanwadhawan7/DocuMind

## Current Status

| Component | Status |
|-----------|--------|
| `POST /upload` (ingestion, chunking, embedding, FAISS index) | Working locally |
| `POST /ask` (retrieval + answer with source pages) | Working locally |
| Streamlit UI | Included, see note below |
| Hosted demo on Render | Limited, see note below |
| MLflow, Prometheus, Grafana | Included in Docker Compose |

**Hosted demo (Render free tier):** the hosted API (https://documind-api-uha8.onrender.com/docs) is limited. The free tier has little memory, and the original version depended on the Hugging Face Inference API for embeddings, whose credits ran out. Embeddings now run locally (see below), but I have not redeployed the hosted version, so some endpoints there may fail.

**Streamlit UI:** `streamlit_app.py` provides a chat interface on top of the API. After switching to local embeddings, I verified the system through the FastAPI endpoints (Swagger UI at `/docs`). The Streamlit container did not respond on my Windows machine after the rebuild (connection reset), and I have not yet debugged it. The API is the verified path.

## What it does

- Upload a PDF: it is split into chunks and indexed in FAISS.
- Ask a question: the most relevant chunks are retrieved and an LLM answers from them only.
- Each answer returns the source page numbers and the response latency.

Example response from `POST /ask`:

```json
{
  "answer": "...",
  "source_pages": [1],
  "latency_seconds": 0.94
}
```

## How it works

1. **Ingestion:** PDF text is extracted per page with `PyPDFLoader`, so page numbers stay in the metadata.
2. **Chunking:** `RecursiveCharacterTextSplitter` (chunk size and overlap are set in `app/ingest.py`).
3. **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, running locally on CPU. No API key or credits needed. The model downloads once (about 90 MB) on first use.
4. **Retrieval:** FAISS similarity search over the chunk embeddings.
5. **Generation:** an open model served by Groq (`openai/gpt-oss-20b`), prompted with the retrieved context. The earlier `llama-3.1-8b-instant` was retired by Groq, so the model was replaced.
6. **Citations:** source pages come from the chunk metadata.

## Tech Stack

| Layer | Technology |
|-------|------------|
| AI Pipeline | LangChain, FAISS, sentence-transformers, Groq (`gpt-oss-20b`) |
| Backend API | FastAPI, Python |
| Experiment Tracking | MLflow |
| Monitoring | Prometheus + Grafana |
| Frontend | Streamlit |
| Containerisation | Docker Compose |
| CI/CD | GitHub Actions |

## Architecture

```
User → Streamlit UI → FastAPI → FAISS vector search → LLM (Groq) → Answer + source pages
                         ↓
        MLflow (experiment tracking)
        Prometheus (metrics) → Grafana (dashboards)
```

## Run locally (without Docker)

```bash
git clone https://github.com/aaryanwadhawan7/DocuMind
cd DocuMind
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env           # add your GROQ_API_KEY
uvicorn app.main:app --port 8000
```

Open http://localhost:8000/docs, then:

1. `POST /upload`: choose a PDF that has selectable text (scanned image-only PDFs have no extractable text).
2. `POST /ask`: ask a question about it.

## Run with Docker Compose

```bash
docker compose up --build
```

The first build is slow because `torch` is large. Services: API (8000), Streamlit (8501), MLflow (5000), Prometheus (9090), Grafana (3000).

## Limitations and next steps

- No evaluation set yet. Planned: a small public-domain sample PDF, test questions with expected answers and pages, and retrieval hit-rate results logged to MLflow.
- Scanned PDFs are not supported (no OCR).
- Missing-evidence handling (refusing to answer when retrieval is weak) is not yet implemented as a similarity threshold.
- Redeploy the hosted version with the local-embeddings setup on a host with more memory.