from dataclasses import dataclass, replace

from fastapi.testclient import TestClient

import backend.main as api
from rag_assistant.agents import Citation, RAGResult
from rag_assistant.service import IndexStatus


class FakeService:
    def get_status(self):
        return IndexStatus(index_ready=True, total_documents=1, total_chunks=2)

    def ask(self, query):
        return RAGResult(
            answer=f"Answered: {query}",
            sources=["policy.txt (Page 1)"],
            rewritten_query=query,
            query_variants=[query],
            classifier_label="IN_SCOPE",
            verifier_passed=True,
            confidence=0.9,
            citations=[
                Citation(
                    marker="[1]",
                    source="policy.txt",
                    page="1",
                    excerpt="Relevant excerpt",
                    retrieval_score=0.9,
                )
            ],
            retrieved_chunks=1,
        )

    def evaluate(self, queries):
        return [self.ask(query) for query in queries]


def test_api_ask_and_evaluate(monkeypatch):
    monkeypatch.setattr(api, "service", FakeService())
    client = TestClient(api.app)

    ask_response = client.post("/ask", json={"query": "policy?"})
    assert ask_response.status_code == 200
    assert ask_response.json()["confidence"] == 0.9
    assert ask_response.json()["citations"][0]["source"] == "policy.txt"

    eval_response = client.post("/evaluate", json={"queries": ["one", "two"]})
    assert eval_response.status_code == 200
    assert eval_response.json()["summary"]["total"] == 2


def test_index_rejects_large_upload(monkeypatch):
    monkeypatch.setattr(api, "settings", replace(api.settings, max_upload_mb=0))
    client = TestClient(api.app)

    response = client.post(
        "/index",
        files={"files": ("large.txt", b"x", "text/plain")},
    )

    assert response.status_code == 413
