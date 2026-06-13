from __future__ import annotations

import os
from typing import Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

from rag_assistant.config import get_settings
from rag_assistant.pipeline import build_pipeline_from_uploads, try_load_existing_pipeline

load_dotenv(override=True)
settings = get_settings()

USE_BACKEND_API = os.getenv("APP_USE_BACKEND_API", "false").strip().lower() == "true"
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000").rstrip("/")


def _api_url(path: str) -> str:
    return f"{BACKEND_API_URL}{path}"


def _backend_status() -> Optional[Dict]:
    try:
        response = requests.get(_api_url("/status"), timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _build_index_via_api(uploaded_files) -> Dict:
    files_payload = [
        ("files", (file_obj.name, file_obj.getvalue(), "application/octet-stream"))
        for file_obj in uploaded_files
    ]
    response = requests.post(
        _api_url("/index"),
        files=files_payload,
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def _ask_via_api(query: str) -> Dict:
    response = requests.post(
        _api_url("/ask"),
        json={"query": query},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def _init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict[str, str]] = []

    if USE_BACKEND_API:
        st.session_state.artifacts = None
        return

    if "artifacts" not in st.session_state:
        st.session_state.artifacts = try_load_existing_pipeline(settings)


def _render_history() -> None:
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])


def _format_assistant_message(result: Dict) -> str:
    lines = [result.get("answer", "Not found in document.")]
    lines.append("")
    lines.append(f"**Confidence**: `{result.get('confidence', 0.0)}`")
    sources = result.get("sources", [])
    if sources:
        lines.append("")
        lines.append("**Sources**")
        for src in sources:
            lines.append(f"- {src}")
    citations = result.get("citations", [])
    if citations:
        lines.append("")
        lines.append("**Citation Evidence**")
        for citation in citations[:5]:
            lines.append(
                "- "
                f"{citation.get('marker')} {citation.get('source')} "
                f"(Page {citation.get('page')}, score {citation.get('retrieval_score')}) "
                f"- {citation.get('excerpt')}"
            )
    lines.append("")
    lines.append(f"_Rewritten query_: `{result.get('rewritten_query', '')}`")
    query_variants = result.get("query_variants", [])
    if query_variants:
        lines.append(f"_Query variants_: `{'; '.join(query_variants)}`")
    lines.append(f"_Verifier passed_: `{result.get('verifier_passed', False)}`")
    return "\n".join(lines)


def _format_local_assistant_message(result) -> str:
    lines = [result.answer]
    lines.append("")
    lines.append(f"**Confidence**: `{result.confidence}`")
    if result.sources:
        lines.append("")
        lines.append("**Sources**")
        for src in result.sources:
            lines.append(f"- {src}")
    if result.citations:
        lines.append("")
        lines.append("**Citation Evidence**")
        for citation in result.citations[:5]:
            lines.append(
                "- "
                f"{citation.marker} {citation.source} "
                f"(Page {citation.page}, score {citation.retrieval_score}) "
                f"- {citation.excerpt}"
            )
    lines.append("")
    lines.append(f"_Rewritten query_: `{result.rewritten_query}`")
    if result.query_variants:
        lines.append(f"_Query variants_: `{'; '.join(result.query_variants)}`")
    lines.append(f"_Verifier passed_: `{result.verifier_passed}`")
    return "\n".join(lines)


def _show_mode_banner() -> None:
    if USE_BACKEND_API:
        st.caption(f"Mode: Backend API (`{BACKEND_API_URL}`)")
    else:
        st.caption("Mode: Local (in-process RAG)")


def main() -> None:
    st.set_page_config(page_title="RAG Document Assistant", layout="wide")
    _init_state()

    st.title("RAG Document Assistant")
    st.caption(
        "Grounded Q&A with hybrid retrieval, citations, confidence scoring, and provider-switchable LLMs"
    )
    _show_mode_banner()

    with st.sidebar:
        st.subheader("Document Indexing")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        if st.button("Build / Rebuild Index", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one supported document.")
            else:
                with st.spinner("Indexing documents..."):
                    try:
                        if USE_BACKEND_API:
                            response = _build_index_via_api(uploaded_files)
                            st.success(
                                "Index ready. Documents: "
                                f"{response.get('total_documents', 0)}, "
                                f"Chunks: {response.get('total_chunks', 0)}"
                            )
                        else:
                            artifacts = build_pipeline_from_uploads(uploaded_files, settings)
                            st.session_state.artifacts = artifacts
                            st.success(
                                f"Index ready. Documents: {artifacts.total_documents}, "
                                f"Chunks: {artifacts.total_chunks}"
                            )
                    except Exception as exc:
                        st.error(f"Indexing failed: {exc}")

        if USE_BACKEND_API:
            status = _backend_status()
            if status is None:
                st.error("Backend unreachable. Start FastAPI service and try again.")
            else:
                st.caption(
                    f"Provider: `{status.get('provider')}` | Model: `{status.get('model')}` | "
                    f"Embeddings: `{status.get('embeddings_model')}`"
                )
                if status.get("index_ready"):
                    st.info(
                        "RAG pipeline is ready (backend). "
                        f"Documents: {status.get('total_documents', 0)}, "
                        f"Chunks: {status.get('total_chunks', 0)}"
                    )
                else:
                    st.info("Build an index to start asking questions.")
        else:
            st.caption(
                f"Provider: `{settings.llm_provider}` | Model: `{settings.ollama_model if settings.llm_provider == 'ollama' else settings.llm_provider}` | "
                f"Embeddings: `{settings.embeddings_model_name}`"
            )
            if st.session_state.artifacts:
                st.info(
                    "RAG pipeline is ready. "
                    f"Documents: {st.session_state.artifacts.total_documents}, "
                    f"Chunks: {st.session_state.artifacts.total_chunks}"
                )
            else:
                st.info("Build an index to start asking questions.")

    _render_history()

    question = st.chat_input("Ask a question from uploaded documents...")
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if USE_BACKEND_API:
                    response = _ask_via_api(question)
                    response_text = _format_assistant_message(response)
                else:
                    if not st.session_state.artifacts:
                        response_text = "No index found. Please upload documents and build the index first."
                    else:
                        result = st.session_state.artifacts.rag.ask(question)
                        response_text = _format_local_assistant_message(result)
            except Exception as exc:
                response_text = (
                    "RAG pipeline error. Ensure backend/Ollama is running and the model is available.\n\n"
                    f"Details: `{exc}`"
                )
            st.markdown(response_text)

    st.session_state.chat_history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()

