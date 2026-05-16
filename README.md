# DocuMind — AI Document Q&A

Upload any PDF and ask questions in plain English. Built as a production-grade MLOps project.

**Live Demo:** https://aaryanwadhawan7-documind-streamlit-app-mgozal.streamlit.app/  
**API Docs:**  https://documind-api-uha8.onrender.com/docs <br>
**GitHub:**    https://github.com/aaryanwadhawan7/DocuMind

## What it does
- Upload a PDF → automatically chunked and indexed
- Ask questions → answers grounded in the document with source page citations
- Full MLOps pipeline with experiment tracking and monitoring

## Tech Stack
| Layer               | Technology                                                |
|---------------------|-----------------------------------------------------------|
| AI Pipeline         | LangChain, FAISS, sentence-transformers, LLaMA 3 via Groq |
| Backend API         | FastAPI, Python                                           |
| Experiment Tracking | MLflow                                                    |
| Monitoring          | Prometheus + Grafana                                      |
| Frontend            | Streamlit                                                 |
| Containerisation    | Docker Compose                                            |
| CI/CD               | GitHub Actions                                            |
| Deployment          | Render                                                    |

## Architecture
User → Streamlit UI → FastAPI → FAISS vector search → LLaMA 3 → Answer
↓
MLflow (experiment tracking)
Prometheus (metrics scraping)
Grafana (live dashboards)

## Run locally
```bash
git clone https://github.com/YOUR_USERNAME/documind
cd documind
cp .env.example .env  # add your GROQ_API_KEY
docker compose up --build
```

Open `localhost:8501` for the chat interface.
