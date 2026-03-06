from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    embeddings_model_name: str = os.getenv(
        "EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "8"))
    retrieval_fetch_k: int = int(os.getenv("RETRIEVAL_FETCH_K", "24"))
    retrieval_lambda_mult: float = float(os.getenv("RETRIEVAL_LAMBDA_MULT", "0.35"))
    max_context_docs: int = int(os.getenv("MAX_CONTEXT_DOCS", "8"))

    vectorstore_dir: Path = Path(os.getenv("VECTORSTORE_DIR", ".rag_index"))


def get_settings() -> Settings:
    return Settings()
