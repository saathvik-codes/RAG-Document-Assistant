from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List

from langchain_core.documents import Document


@dataclass
class SourceStats:
    source: str
    doc_type: str
    pages: int
    chunks: int


@dataclass
class IndexManifest:
    created_at: str
    embeddings_model: str
    chunk_size: int
    chunk_overlap: int
    total_documents: int
    total_chunks: int
    content_hash: str
    sources: List[SourceStats] = field(default_factory=list)


def _hash_chunks(chunks: Iterable[Document]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.page_content.encode("utf-8", errors="ignore"))
        digest.update(json.dumps(chunk.metadata, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def build_manifest(
    docs: List[Document],
    chunks: List[Document],
    embeddings_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> IndexManifest:
    source_pages: dict[str, set[Any]] = {}
    source_types: dict[str, str] = {}
    source_chunks: dict[str, int] = {}

    for doc in docs:
        source = str(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "N/A")
        source_pages.setdefault(source, set()).add(page)
        source_types[source] = str(doc.metadata.get("doc_type", "unknown"))

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "Unknown"))
        source_chunks[source] = source_chunks.get(source, 0) + 1

    sources = [
        SourceStats(
            source=source,
            doc_type=source_types.get(source, "unknown"),
            pages=len(pages),
            chunks=source_chunks.get(source, 0),
        )
        for source, pages in sorted(source_pages.items())
    ]

    return IndexManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        embeddings_model=embeddings_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        total_documents=len(docs),
        total_chunks=len(chunks),
        content_hash=_hash_chunks(chunks),
        sources=sources,
    )


def save_manifest(manifest: IndexManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(manifest)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> IndexManifest | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sources"] = [SourceStats(**item) for item in payload.get("sources", [])]
    return IndexManifest(**payload)
