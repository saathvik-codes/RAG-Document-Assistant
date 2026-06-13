from langchain_core.documents import Document

from rag_assistant.config import Settings
from rag_assistant.retrieval import HybridRetriever


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def as_retriever(self, **_kwargs):
        return self

    def invoke(self, _query):
        return [self.docs[1]]


def test_hybrid_retriever_promotes_keyword_match(tmp_path):
    docs = [
        Document(
            page_content="The remote work policy allows two work from home days.",
            metadata={"source": "policy.txt", "page": 1},
        ),
        Document(
            page_content="The cafeteria menu includes tea and snacks.",
            metadata={"source": "menu.txt", "page": 1},
        ),
    ]
    settings = Settings(
        vectorstore_dir=tmp_path / "idx",
        manifest_path=tmp_path / "idx" / "manifest.json",
        retrieval_k=1,
        keyword_retrieval_k=2,
        max_context_docs=2,
    )

    retriever = HybridRetriever(FakeVectorStore(docs), docs, settings)
    results = retriever.invoke("remote work policy")

    assert results[0].metadata["source"] == "policy.txt"
    assert results[0].metadata["retrieval_score"] > 0


def test_hybrid_retriever_uses_source_affinity_for_survey_questions(tmp_path):
    docs = [
        Document(
            page_content="AI governance includes environmental and organizational layers.",
            metadata={"source": "ai_governance_hourglass_model.pdf", "page": 1},
        ),
        Document(
            page_content="We surveyed 524 AI/ML researchers about AI safety and pre-publication review.",
            metadata={"source": "ml_researcher_ai_ethics_survey.pdf", "page": 1},
        ),
    ]
    settings = Settings(
        vectorstore_dir=tmp_path / "idx",
        manifest_path=tmp_path / "idx" / "manifest.json",
        retrieval_k=1,
        keyword_retrieval_k=2,
        max_context_docs=2,
    )

    retriever = HybridRetriever(FakeVectorStore(docs), docs, settings)
    results = retriever.invoke("How many AI/ML researchers were surveyed?")

    assert results[0].metadata["source"] == "ml_researcher_ai_ethics_survey.pdf"
