from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from rag_assistant.agents import AgenticRAGPipeline
from rag_assistant.config import Settings
from rag_assistant.document_ingestion import chunk_documents, load_uploaded_documents
from rag_assistant.vector_store import build_and_persist_faiss, load_existing_faiss


@dataclass
class PipelineArtifacts:
    total_documents: int
    total_chunks: int
    vectorstore: Any
    rag: AgenticRAGPipeline


def _build_retriever(vectorstore: Any, settings: Settings):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retrieval_k,
            "fetch_k": max(settings.retrieval_fetch_k, settings.retrieval_k),
            "lambda_mult": settings.retrieval_lambda_mult,
        },
    )


def build_pipeline_from_uploads(uploaded_files: List[Any], settings: Settings) -> PipelineArtifacts:
    docs = load_uploaded_documents(uploaded_files)
    if not docs:
        raise ValueError("No supported documents found. Upload PDF, DOCX, or TXT files.")

    chunks = chunk_documents(
        docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise ValueError("Document chunking failed. Check file content and try again.")

    vectorstore = build_and_persist_faiss(
        chunks=chunks,
        embeddings_model_name=settings.embeddings_model_name,
        persist_dir=settings.vectorstore_dir,
    )

    retriever = _build_retriever(vectorstore, settings)
    rag = AgenticRAGPipeline(retriever=retriever, settings=settings)
    return PipelineArtifacts(
        total_documents=len(docs),
        total_chunks=len(chunks),
        vectorstore=vectorstore,
        rag=rag,
    )


def try_load_existing_pipeline(settings: Settings) -> PipelineArtifacts | None:
    vectorstore = load_existing_faiss(
        embeddings_model_name=settings.embeddings_model_name,
        persist_dir=settings.vectorstore_dir,
    )
    if vectorstore is None:
        return None

    retriever = _build_retriever(vectorstore, settings)
    rag = AgenticRAGPipeline(retriever=retriever, settings=settings)
    return PipelineArtifacts(
        total_documents=0,
        total_chunks=0,
        vectorstore=vectorstore,
        rag=rag,
    )
