# RAG Document Assistant (Enterprise-Grade Starter)

RAG chatbot for enterprise documents with source-grounded answers, agentic workflow, and scalable deployment.

## Tech Stack

- Python
- FastAPI (backend API)
- Streamlit (UI)
- LangChain
- HuggingFace embeddings (`sentence-transformers`)
- FAISS vector database
- Llama 3 via Ollama
- Docker / Docker Compose

## Features Implemented

- Document ingestion for `PDF`, `DOCX`, and `TXT`
- Chunking with overlap and metadata (`source`, `page`, `doc_type`)
- Section-aware chunking for policy-style documents
- FAISS indexing with local persistence
- Grounded retrieval + answer generation
- Source-aware answers (`Document.pdf (Page X)`)
- Agentic pipeline:
  - Query classifier
  - Query rewriter
  - Answer generator
  - Hallucination checker
- Backend API endpoints:
  - `GET /health`
  - `GET /status`
  - `POST /index`
  - `POST /ask`
- Streamlit supports:
  - Local in-process mode
  - Backend API mode

## Project Structure

```text
.
|-- app.py
|-- backend/
|   `-- main.py
|-- rag_assistant/
|   |-- agents.py
|   |-- config.py
|   |-- document_ingestion.py
|   |-- pipeline.py
|   |-- service.py
|   `-- vector_store.py
|-- Dockerfile.backend
|-- Dockerfile.frontend
|-- docker-compose.yml
|-- requirements.txt
|-- .env.example
`-- README.md
```

## Local Run (Without Docker)

1. Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

2. Start Ollama and pull model:

```powershell
ollama pull llama3
ollama serve
```

3. Start backend:

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

4. Start frontend (API mode):

```powershell
$env:APP_USE_BACKEND_API="true"
$env:BACKEND_API_URL="http://localhost:8000"
streamlit run app.py
```

## Docker Run (Recommended)

```powershell
docker compose up --build
```

Services:
- Streamlit UI: `http://localhost:8501`
- FastAPI backend: `http://localhost:8000`
- Ollama: `http://localhost:11434`

Stop:

```powershell
docker compose down
```

## Environment Variables

See `.env.example`:

- `EMBEDDINGS_MODEL_NAME`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVAL_K`
- `RETRIEVAL_FETCH_K`
- `RETRIEVAL_LAMBDA_MULT`
- `MAX_CONTEXT_DOCS`
- `VECTORSTORE_DIR`
- `APP_USE_BACKEND_API`
- `BACKEND_API_URL`

## Credentials / Keys

Default setup needs no external API key (local Ollama + local embeddings).

Optional only if you switch providers:
- `HUGGINGFACEHUB_API_TOKEN`
- `OPENAI_API_KEY`

## Interview Positioning

- "RAG for grounding, fine-tuning for style and reasoning."
- Current build demonstrates grounding + traceability + agentic workflow + deployability.
