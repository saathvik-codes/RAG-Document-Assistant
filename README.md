# RAG Document Assistant

Production-oriented RAG assistant for source-grounded enterprise document Q&A.

## Verified results

Re-run locally (Ollama `llama3.1:8b`, hashing-embeddings fallback) rather
than asserted:

- **16/16** automated tests pass (`pytest tests/`)
- **11/11** eval cases pass (`python scripts/run_eval.py`) — citation-grounded
  answers to in-scope questions, correct refusal on out-of-scope ones
  (capital of France, latest sports score)

Getting to 11/11 surfaced two real bugs worth naming, since "we ran the
eval and it passed" is a weaker claim than "here's what broke and what
fixing it actually looked like":
- The query classifier had no idea what documents were actually indexed,
  so it judged relevance on the question's wording alone — two legitimate,
  in-scope questions ("What does the hourglass metaphor represent?", "How
  did respondents feel about lethal autonomous weapons?") got misclassified
  as out-of-scope and refused. Fix: the classifier prompt now includes the
  actual indexed source filenames, so it has something real to judge
  relevance against (`rag_assistant/agents.py`).
- The eval harness checked the answer against a single hardcoded refusal
  string (`"not found in document."`), but the classifier's refusal path
  returns a different sentinel (`"Please ask a question related to the
  uploaded documents."`) — so both negative test cases (questions that
  *should* be refused) were failing even though the system was behaving
  correctly. Same bug existed in the `/evaluate` endpoint's summary stats.
  Fixed by centralizing both refusal sentinels in one place
  (`rag_assistant.agents.REFUSAL_ANSWERS`) instead of three separate
  hardcoded string literals that had drifted out of sync.

No hosted live demo: this isn't a static dashboard, it's a real inference
API, and Ollama on free-tier CPU hosting is too slow to be a usable demo —
running a hosted LLM provider (OpenAI/Anthropic/Gemini) instead is one env
var away (`LLM_PROVIDER`, see below) but needs a paid API key. Runs fully
locally via Docker Compose with no API key at all.

## What It Does

- Ingests `PDF`, `DOCX`, and `TXT` documents
- Cleans text and preserves metadata such as source, page, and document type
- Splits documents with section-aware chunking
- Builds a FAISS semantic index with HuggingFace embeddings
- Adds lightweight keyword/BM25-style scoring for hybrid retrieval
- Generates multiple retrieval query variants
- Answers only from retrieved context
- Verifies answers with a hallucination-checking agent
- Returns source citations, excerpts, retrieval scores, confidence, and diagnostics
- Persists an index manifest for production observability
- Exposes FastAPI endpoints and a Streamlit UI

## Architecture

```text
Upload -> Loader -> Cleaner -> Chunker -> Embeddings -> FAISS Index
                                                    |
Question -> Classifier -> Query Variants -> Hybrid Retriever
                                                    |
Context -> Answer Agent -> Verifier -> Citations + Confidence
```

## Tech Stack

- Python
- FastAPI
- Streamlit
- LangChain
- HuggingFace sentence-transformer embeddings
- Dependency-light hashing embeddings fallback for offline smoke tests
- FAISS vector database
- Ollama by default
- Optional hosted LLM providers: OpenAI, Anthropic, Google Gemini
- Docker / Docker Compose

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
|   |-- evaluation.py
|   |-- index_manifest.py
|   |-- llm.py
|   |-- pipeline.py
|   |-- retrieval.py
|   |-- service.py
|   `-- vector_store.py
|-- Dockerfile.backend
|-- Dockerfile.frontend
|-- docker-compose.yml
|-- requirements.txt
|-- .env.example
`-- README.md
```

## API Endpoints

- `GET /health` - liveness check
- `GET /status` - model, embedding, index, and manifest status
- `POST /index` - upload and index documents
- `POST /ask` - ask one grounded question
- `POST /evaluate` - run a batch of questions and return quality metrics

## Local Run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Start Ollama:

```powershell
ollama pull llama3
ollama serve
```

Start backend:

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Start frontend in API mode:

```powershell
$env:APP_USE_BACKEND_API="true"
$env:BACKEND_API_URL="http://localhost:8000"
streamlit run app.py
```

## Model Providers

Default local mode:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3
```

If your local ML stack cannot import `sentence_transformers`, use the built-in fallback:

```env
EMBEDDINGS_MODEL_NAME=hash
```

Hosted provider examples:

```env
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=...
```

```env
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-3-5-haiku-latest
ANTHROPIC_API_KEY=...
```

```env
LLM_PROVIDER=google
GOOGLE_MODEL=gemini-1.5-flash
GOOGLE_API_KEY=...
```

## Docker Run

```powershell
docker compose up --build
```

Services:

- Streamlit UI: `http://localhost:8501`
- FastAPI backend: `http://localhost:8000`
- Ollama: `http://localhost:11434`

## Hosted Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for the Render Blueprint flow, production environment variables, smoke tests, and hosted LLM provider setup.

## Production Features Added

- Provider-switchable LLM layer
- Hybrid semantic + keyword retrieval
- Source-aware reranking for multi-document collections
- Multi-query retrieval expansion
- Deterministic evidence composition for high-precision factual questions
- Citation objects with excerpts and retrieval scores
- Answer confidence scoring
- Strict hallucination verifier
- Index manifest with document/chunk/source stats
- Batch evaluation endpoint
- API contracts suitable for frontend, testing, and monitoring

## Evaluation

Use the included public-source sample documents and eval suite before demos.

Index the two clean PDF documents:

```powershell
curl.exe -s -X POST http://127.0.0.1:8010/index `
  -F "files=@sample_documents/ai_governance_hourglass_model.pdf" `
  -F "files=@sample_documents/ml_researcher_ai_ethics_survey.pdf"
```

Run the regression eval:

```powershell
python scripts/run_eval.py --base-url http://127.0.0.1:8010
```

The suite checks grounded answers, required evidence terms, confidence, verifier status, and refusal behavior for unsupported questions.

## Interview Positioning

This project demonstrates a production RAG architecture, not just a chatbot:

- ingestion and chunking pipeline
- vector indexing and persistence
- retrieval quality improvements
- source-aware reranking
- grounded answer generation
- hallucination control
- deterministic evaluation suite
- citations and confidence
- API-first deployment
- evaluation readiness
