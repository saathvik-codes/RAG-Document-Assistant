from rag_assistant.config import Settings
from rag_assistant.pipeline import build_pipeline_from_uploads


class InMemoryUpload:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


class FakeEmbeddings:
    def _embed(self, text):
        tokens = text.lower().split()
        return [
            float(len(tokens)),
            float(text.lower().count("policy")),
            float(text.lower().count("remote")),
            float(text.lower().count("security")),
        ]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def test_pipeline_builds_faiss_index_and_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr("rag_assistant.vector_store.build_embeddings", lambda _name: FakeEmbeddings())
    upload = InMemoryUpload(
        "policy.txt",
        b"Remote work policy allows two days per week.\nSecurity policy requires MFA.",
    )
    settings = Settings(
        vectorstore_dir=tmp_path / "idx",
        manifest_path=tmp_path / "idx" / "manifest.json",
        chunk_size=120,
        chunk_overlap=20,
    )

    artifacts = build_pipeline_from_uploads([upload], settings)

    assert artifacts.total_documents == 1
    assert artifacts.total_chunks >= 1
    assert artifacts.manifest is not None
    assert artifacts.manifest.sources[0].source == "policy.txt"
    assert (tmp_path / "idx" / "index.faiss").exists()
    assert (tmp_path / "idx" / "manifest.json").exists()
