# DocuMind

- RAG Pipeline
```
PDF -> Chunks -> Embeddings -> FAISS Index -> LLM -> Answer
```

- MLflow : 
  - Analytics for ur AI system
  - Appn. (Function) : Takes input, processes it and gives output && MLflow keeps track of it

- Github Actions: 
  - Pipeline
  - Whenever we push or pull-request then test run automatically then creates new docker image
              
#### Problem Encountered
- Inside docker networking, containers find each other by looking at their service names like mlflow.
- This works perfectly inside docker compose.
- But when running tests in GitHub Actions (Works on a linux-machine), it fails test when looking mlflow
- Solution: The solution is to only connect to MLflow when actually needed, not when the file loads. We use an environment variable to control this.

- Prometheus (Sensors) && Grafana (Dashboard)