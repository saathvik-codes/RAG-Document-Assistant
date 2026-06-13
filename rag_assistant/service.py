from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import List, Sequence

from rag_assistant.agents import RAGResult
from rag_assistant.config import Settings
from rag_assistant.index_manifest import IndexManifest
from rag_assistant.pipeline import build_pipeline_from_uploads, try_load_existing_pipeline


class InMemoryUpload:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


@dataclass
class IndexStatus:
    index_ready: bool
    total_documents: int
    total_chunks: int
    manifest: IndexManifest | None = None


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = RLock()
        self._artifacts = try_load_existing_pipeline(settings)
        self._total_documents = self._artifacts.total_documents if self._artifacts else 0
        self._total_chunks = self._artifacts.total_chunks if self._artifacts else 0
        self._answer_cache: dict[str, RAGResult] = {}

    def build_index(self, files: Sequence[tuple[str, bytes]]) -> IndexStatus:
        uploads = [InMemoryUpload(name=name, content=content) for name, content in files]
        artifacts = build_pipeline_from_uploads(uploads, self.settings)

        with self._lock:
            self._artifacts = artifacts
            self._total_documents = artifacts.total_documents
            self._total_chunks = artifacts.total_chunks
            self._answer_cache.clear()

        return IndexStatus(
            index_ready=True,
            total_documents=self._total_documents,
            total_chunks=self._total_chunks,
        )

    def ask(self, query: str) -> RAGResult:
        cache_key = query.strip().lower()
        with self._lock:
            artifacts = self._artifacts
            cached = self._answer_cache.get(cache_key)
        if artifacts is None:
            raise RuntimeError("No vector index found. Build the index first.")
        if cached is not None:
            return cached
        result = artifacts.rag.ask(query)
        with self._lock:
            self._answer_cache[cache_key] = result
        return result

    def get_status(self) -> IndexStatus:
        with self._lock:
            manifest = self._artifacts.manifest if self._artifacts else None
            return IndexStatus(
                index_ready=self._artifacts is not None,
                total_documents=self._total_documents,
                total_chunks=self._total_chunks,
                manifest=manifest,
            )

    def evaluate(self, queries: Sequence[str]) -> List[RAGResult]:
        return [self.ask(query) for query in queries if query.strip()]

