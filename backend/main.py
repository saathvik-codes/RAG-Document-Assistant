from __future__ import annotations

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag_assistant.config import get_settings
from rag_assistant.document_ingestion import SUPPORTED_EXTENSIONS
from rag_assistant.service import RAGService

load_dotenv()
settings = get_settings()
service = RAGService(settings)

app = FastAPI(
    title="RAG Document Assistant API",
    version="0.1.0",
    description="Enterprise-grade RAG backend for document-grounded Q&A.",
)


class StatusResponse(BaseModel):
    index_ready: bool
    total_documents: int
    total_chunks: int
    model: str
    embeddings_model: str


class IndexResponse(BaseModel):
    message: str
    index_ready: bool
    total_documents: int
    total_chunks: int


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    rewritten_query: str
    classifier_label: str
    verifier_passed: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    current = service.get_status()
    return StatusResponse(
        index_ready=current.index_ready,
        total_documents=current.total_documents,
        total_chunks=current.total_chunks,
        model=settings.ollama_model,
        embeddings_model=settings.embeddings_model_name,
    )


@app.post("/index", response_model=IndexResponse)
async def index_documents(files: List[UploadFile] = File(...)) -> IndexResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    valid_files: List[tuple[str, bytes]] = []
    for upload in files:
        filename = upload.filename or ""
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue
        content = await upload.read()
        if not content:
            continue
        valid_files.append((filename, content))

    if not valid_files:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"No supported non-empty files found. Supported: {supported}",
        )

    try:
        status_info = service.build_index(valid_files)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    return IndexResponse(
        message="Index built successfully.",
        index_ready=status_info.index_ready,
        total_documents=status_info.total_documents,
        total_chunks=status_info.total_chunks,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    try:
        result = service.ask(payload.query)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}") from exc

    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        rewritten_query=result.rewritten_query,
        classifier_label=result.classifier_label,
        verifier_passed=result.verifier_passed,
    )

