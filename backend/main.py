from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag_assistant.config import get_settings
from rag_assistant.document_ingestion import SUPPORTED_EXTENSIONS
from rag_assistant.evaluation import summarize_results
from rag_assistant.llm import active_model_name
from rag_assistant.service import RAGService

load_dotenv(override=True)
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
    provider: str
    model: str
    embeddings_model: str
    manifest: Optional[dict] = None


class IndexResponse(BaseModel):
    message: str
    index_ready: bool
    total_documents: int
    total_chunks: int


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)


class CitationResponse(BaseModel):
    marker: str
    source: str
    page: str
    excerpt: str
    retrieval_score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    rewritten_query: str
    query_variants: List[str]
    classifier_label: str
    verifier_passed: bool
    confidence: float
    citations: List[CitationResponse]
    retrieved_chunks: int


class EvaluateRequest(BaseModel):
    queries: List[str] = Field(min_length=1, max_length=50)


class EvaluationSummaryResponse(BaseModel):
    total: int
    answered: int
    not_found: int
    verifier_pass_rate: float
    average_confidence: float


class EvaluateResponse(BaseModel):
    summary: EvaluationSummaryResponse
    results: List[AskResponse]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
    }


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    current = service.get_status()
    return StatusResponse(
        index_ready=current.index_ready,
        total_documents=current.total_documents,
        total_chunks=current.total_chunks,
        provider=settings.llm_provider,
        model=active_model_name(settings),
        embeddings_model=settings.embeddings_model_name,
        manifest=asdict(current.manifest) if current.manifest else None,
    )


@app.post("/index", response_model=IndexResponse)
async def index_documents(files: List[UploadFile] = File(...)) -> IndexResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    valid_files: List[tuple[str, bytes]] = []
    total_bytes = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024
    for upload in files:
        filename = upload.filename or ""
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue
        content = await upload.read()
        if not content:
            continue
        total_bytes += len(content)
        if total_bytes > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload too large. Max total upload size is {settings.max_upload_mb} MB.",
            )
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

    return _to_ask_response(result)


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(payload: EvaluateRequest) -> EvaluateResponse:
    try:
        results = service.evaluate(payload.queries)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    summary = summarize_results(results)
    return EvaluateResponse(
        summary=EvaluationSummaryResponse(**summary.__dict__),
        results=[_to_ask_response(result) for result in results],
    )


def _to_ask_response(result) -> AskResponse:
    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        rewritten_query=result.rewritten_query,
        query_variants=result.query_variants,
        classifier_label=result.classifier_label,
        verifier_passed=result.verifier_passed,
        confidence=result.confidence,
        citations=[
            CitationResponse(
                marker=item.marker,
                source=item.source,
                page=item.page,
                excerpt=item.excerpt,
                retrieval_score=item.retrieval_score,
            )
            for item in result.citations
        ],
        retrieved_chunks=result.retrieved_chunks,
    )

