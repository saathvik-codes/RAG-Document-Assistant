from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import hashlib

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class HashingEmbeddings(Embeddings):
    """Dependency-light fallback embeddings for local smoke tests and offline demos."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def build_embeddings(model_name: str) -> Embeddings:
    if model_name.strip().lower() in {"hash", "hashing", "local-hash"}:
        return HashingEmbeddings()
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        print(f"Falling back to local hashing embeddings: {exc}")
        return HashingEmbeddings()


def build_and_persist_faiss(
    chunks: List[Document], embeddings_model_name: str, persist_dir: Path
) -> FAISS:
    if not chunks:
        raise ValueError("No chunks were provided for indexing.")

    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = build_embeddings(embeddings_model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(persist_dir))
    return vectorstore


def load_existing_faiss(embeddings_model_name: str, persist_dir: Path) -> Optional[FAISS]:
    index_file = persist_dir / "index.faiss"
    if not index_file.exists():
        return None

    embeddings = build_embeddings(embeddings_model_name)
    return FAISS.load_local(
        str(persist_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def extract_documents(vectorstore: FAISS) -> List[Document]:
    return list(vectorstore.docstore._dict.values())

