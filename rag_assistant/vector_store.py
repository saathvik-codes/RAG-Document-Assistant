from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


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

