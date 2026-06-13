from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "RAG Document Assistant"))
    environment: str = field(default_factory=lambda: os.getenv("APP_ENV", "local"))

    embeddings_model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0")))
    llm_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT_SECONDS", "120")))

    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"))
    google_model: str = field(default_factory=lambda: os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"))

    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))
    retrieval_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_K", "8")))
    retrieval_fetch_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_FETCH_K", "24")))
    retrieval_lambda_mult: float = field(default_factory=lambda: float(os.getenv("RETRIEVAL_LAMBDA_MULT", "0.35")))
    max_context_docs: int = field(default_factory=lambda: int(os.getenv("MAX_CONTEXT_DOCS", "8")))
    multi_query_count: int = field(default_factory=lambda: int(os.getenv("MULTI_QUERY_COUNT", "3")))
    keyword_retrieval_k: int = field(default_factory=lambda: int(os.getenv("KEYWORD_RETRIEVAL_K", "8")))
    min_answer_confidence: float = field(default_factory=lambda: float(os.getenv("MIN_ANSWER_CONFIDENCE", "0.35")))

    vectorstore_dir: Path = field(default_factory=lambda: Path(os.getenv("VECTORSTORE_DIR", ".rag_index")))
    manifest_path: Path = field(
        default_factory=lambda: Path(os.getenv("INDEX_MANIFEST_PATH", ".rag_index/manifest.json"))
    )
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "50")))


def get_settings() -> Settings:
    return Settings()
