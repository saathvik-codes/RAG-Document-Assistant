from dataclasses import dataclass

from rag_assistant.config import Settings
from rag_assistant.service import RAGService


@dataclass
class FakeArtifacts:
    total_documents: int
    total_chunks: int
    manifest: object | None
    rag: object


class CountingRAG:
    def __init__(self):
        self.calls = 0

    def ask(self, query):
        from rag_assistant.agents import RAGResult

        self.calls += 1
        return RAGResult(
            answer=f"answer {query}",
            sources=[],
            rewritten_query=query,
            query_variants=[query],
            classifier_label="IN_SCOPE",
            verifier_passed=True,
            confidence=1.0,
            citations=[],
            retrieved_chunks=1,
        )


def test_service_caches_answers(monkeypatch, tmp_path):
    rag = CountingRAG()

    def fake_loader(_settings):
        return FakeArtifacts(total_documents=1, total_chunks=1, manifest=None, rag=rag)

    monkeypatch.setattr("rag_assistant.service.try_load_existing_pipeline", fake_loader)
    service = RAGService(Settings(vectorstore_dir=tmp_path / "idx"))

    first = service.ask("What is policy?")
    second = service.ask("What is policy?")

    assert first.answer == second.answer
    assert rag.calls == 1
