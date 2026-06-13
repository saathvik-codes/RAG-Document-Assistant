from langchain_core.documents import Document

from rag_assistant.index_manifest import build_manifest, load_manifest, save_manifest


def test_manifest_round_trip(tmp_path):
    docs = [
        Document(page_content="A", metadata={"source": "policy.pdf", "page": 1, "doc_type": "pdf"}),
        Document(page_content="B", metadata={"source": "policy.pdf", "page": 2, "doc_type": "pdf"}),
    ]
    chunks = [
        Document(page_content="remote work policy", metadata={"source": "policy.pdf", "page": 1}),
        Document(page_content="leave policy", metadata={"source": "policy.pdf", "page": 2}),
    ]

    manifest = build_manifest(
        docs=docs,
        chunks=chunks,
        embeddings_model="test-embeddings",
        chunk_size=500,
        chunk_overlap=100,
    )
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)

    loaded = load_manifest(path)

    assert loaded is not None
    assert loaded.total_documents == 2
    assert loaded.total_chunks == 2
    assert loaded.sources[0].source == "policy.pdf"
    assert loaded.sources[0].pages == 2
    assert loaded.sources[0].chunks == 2
