# Deployment Guide

This app is deployment-ready with Docker and a Render Blueprint.

## Recommended Hosted Setup

Deploy two Render web services:

1. `rag-document-assistant-api`
   - Uses `Dockerfile.backend`
   - Exposes FastAPI
   - Health check: `/health`

2. `rag-document-assistant-ui`
   - Uses `Dockerfile.frontend`
   - Exposes Streamlit
   - Calls the backend through `BACKEND_API_URL`

## Required Environment Variables

For hosted deployment, use a hosted LLM provider. Local Ollama is best for local demos, but hosted platforms need a reachable provider.

Minimum recommended production settings:

```env
APP_ENV=production
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=<set in host secret manager>
EMBEDDINGS_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
VECTORSTORE_DIR=/tmp/rag_index
INDEX_MANIFEST_PATH=/tmp/rag_index/manifest.json
```

For the UI service:

```env
APP_USE_BACKEND_API=true
BACKEND_API_URL=https://<your-api-service-url>
```

## Render Deployment

1. Push the repo to GitHub.
2. In Render, create a new Blueprint from this repository.
3. Set `LLM_PROVIDER` and the matching API key.
4. After Render creates the API service, set the UI service `BACKEND_API_URL` to the API URL.
5. Redeploy the UI service.

## Smoke Test

After deployment:

```powershell
curl.exe https://<api-url>/health
curl.exe https://<api-url>/status
```

Then open the UI URL, upload files from `sample_documents/`, build the index, and run:

```text
What are the three layers in the hourglass model of organizational AI governance?
```

Expected answer:

```text
The three layers are environmental, organizational, and AI system.
```

## Local Quality Gate

Before every deploy:

```powershell
python -m pytest -q
python scripts/run_eval.py --base-url http://127.0.0.1:8010
```
