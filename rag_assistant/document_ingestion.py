from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, List

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

SECTION_AWARE_SEPARATORS = [
    r"\n(?=\d{1,2}(?:\.\d+)*\s+[A-Z])",
    r"\n(?=[A-Z][A-Z0-9/&,\- ]{4,80}\n)",
    r"\n{2,}",
    r"\n",
    r"(?<=[.!?])\s+",
    r"\s+",
]

SPACED_LETTERS_PATTERN = re.compile(r"(?<!\w)(?:[A-Za-z]\s+){3,}[A-Za-z](?!\w)")


def _build_loader(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(file_path))
    if suffix == ".docx":
        return Docx2txtLoader(str(file_path))
    if suffix == ".txt":
        return TextLoader(str(file_path), encoding="utf-8")
    raise ValueError(f"Unsupported file type: {suffix}")


def _normalize_metadata(docs: Iterable[Document], filename: str, suffix: str) -> List[Document]:
    normalized_docs: List[Document] = []
    for doc in docs:
        page = doc.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else None
        doc.metadata["source"] = filename
        doc.metadata["doc_type"] = suffix.lstrip(".")
        if page_number is not None:
            doc.metadata["page"] = page_number
        normalized_docs.append(doc)
    return normalized_docs


def _join_spaced_letters(match: re.Match[str]) -> str:
    letters = re.findall(r"[A-Za-z]", match.group(0))
    return "".join(letters)


def _normalize_page_content(text: str) -> str:
    if not text:
        return ""
    collapsed = SPACED_LETTERS_PATTERN.sub(_join_spaced_letters, text)
    collapsed = re.sub(r"[ \t]{2,}", " ", collapsed)
    return collapsed.strip()


def _build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    kwargs = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "separators": SECTION_AWARE_SEPARATORS,
        "is_separator_regex": True,
        "keep_separator": True,
    }
    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(**kwargs)
    except Exception:
        return RecursiveCharacterTextSplitter(**kwargs)


def load_uploaded_documents(uploaded_files: List[Any]) -> List[Document]:
    all_docs: List[Document] = []
    if not uploaded_files:
        return all_docs

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        for file_obj in uploaded_files:
            filename = file_obj.name
            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue

            temp_file = temp_dir_path / filename
            temp_file.write_bytes(file_obj.getvalue())
            loader = _build_loader(temp_file)
            docs = loader.load()
            all_docs.extend(_normalize_metadata(docs, filename, suffix))
    return all_docs


def chunk_documents(
    docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 200
) -> List[Document]:
    if not docs:
        return []

    splitter = _build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    normalized_docs: List[Document] = []
    for doc in docs:
        cleaned = _normalize_page_content(doc.page_content)
        if not cleaned:
            continue
        normalized_docs.append(
            Document(
                page_content=cleaned,
                metadata=dict(doc.metadata),
            )
        )

    return splitter.split_documents(normalized_docs)
