from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from rag_assistant.agents import AgenticRAGPipeline
from rag_assistant.config import Settings
from rag_assistant.document_ingestion import chunk_documents, load_uploaded_documents
from rag_assistant.index_manifest import IndexManifest, build_manifest, load_manifest, save_manifest
from rag_assistant.retrieval import HybridRetriever
from rag_assistant.vector_store import (
    build_and_persist_faiss,
    extract_documents,
    load_existing_faiss,
)


@dataclass
class PipelineArtifacts:
    total_documents: int
    total_chunks: int
    vectorstore: Any
    rag: AgenticRAGPipeline
    manifest: IndexManifest | None = None


def _build_retriever(vectorstore: Any, chunks: List[Any], settings: Settings):
    return HybridRetriever(vectorstore=vectorstore, chunks=chunks, settings=settings)


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

    manifest = build_manifest(
        docs=docs,
        chunks=chunks,
        embeddings_model=settings.embeddings_model_name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    save_manifest(manifest, settings.manifest_path)

    retriever = _build_retriever(vectorstore, chunks, settings)
    rag = AgenticRAGPipeline(retriever=retriever, settings=settings)
    return PipelineArtifacts(
        total_documents=len(docs),
        total_chunks=len(chunks),
        vectorstore=vectorstore,
        rag=rag,
        manifest=manifest,
    )


def try_load_existing_pipeline(settings: Settings) -> PipelineArtifacts | None:
    vectorstore = load_existing_faiss(
        embeddings_model_name=settings.embeddings_model_name,
        persist_dir=settings.vectorstore_dir,
    )
    if vectorstore is None:
        return None

    chunks = extract_documents(vectorstore)
    manifest = load_manifest(settings.manifest_path)
    retriever = _build_retriever(vectorstore, chunks, settings)
    rag = AgenticRAGPipeline(retriever=retriever, settings=settings)
    return PipelineArtifacts(
        total_documents=manifest.total_documents if manifest else 0,
        total_chunks=manifest.total_chunks if manifest else len(chunks),
        vectorstore=vectorstore,
        rag=rag,
        manifest=manifest,
    )
