from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from rag_assistant.agents import AgenticRAGPipeline
from rag_assistant.config import Settings


class FakeRetriever:
    def invoke(self, _query):
        return [
            Document(
                page_content="The policy allows remote work two days per week.",
                metadata={"source": "policy.txt", "page": 3, "retrieval_score": 0.95},
            )
        ]


class LayerRetriever:
    def invoke(self, _query):
        return [
            Document(
                page_content=(
                    "The Hourglass Model of Organizational AI Governance proposes AI governance components. "
                    "The layers are environmental, organizational, and AI system."
                ),
                metadata={"source": "governance.pdf", "page": 5, "retrieval_score": 0.95},
            )
        ]


class HourglassMetaphorRetriever:
    def invoke(self, _query):
        return [
            Document(
                page_content=(
                    "The hourglass metaphor denotes the flow of governance requirements from the "
                    "environmental layer to AI systems through the mediating organizational layer."
                ),
                metadata={"source": "governance.pdf", "page": 5, "retrieval_score": 0.92},
            )
        ]


class HeadOfAIRetriever:
    def invoke(self, _query):
        return [
            Document(
                page_content=(
                    "The organization should establish an organizational role responsible for overseeing "
                    "AI system development and AI system operations. We refer to this organizational role as Head of AI. "
                    "The Head of AI should have sufficient knowledge and authority and resources."
                ),
                metadata={"source": "governance.pdf", "page": 17, "retrieval_score": 0.88},
            )
        ]


class SurveyFindingsRetriever:
    def invoke(self, _query):
        return [
            Document(
                page_content=(
                    "We find that AI/ML researchers place high levels of trust in international organizations "
                    "and scientific organizations to shape the development and use of AI in the public interest. "
                    "While the respondents were overwhelmingly opposed to AI/ML researchers working on lethal "
                    "autonomous weapons, a strong majority of respondents think that AI safety research should "
                    "be prioritized and that ML institutions should conduct pre-publication review to assess "
                    "potential harms."
                ),
                metadata={"source": "survey.pdf", "page": 1, "retrieval_score": 0.9},
            )
        ]


class PdfArtifactTrustRetriever:
    def invoke(self, _query):
        return [
            Document(
                page_content=(
                    "We find that AI/ML researchers place high levels of trust in interna- "
                    "tional organizations and scienti\ufb01c organizations to shape the development "
                    "and use of AI in the public interest."
                ),
                metadata={"source": "survey.pdf", "page": 1, "retrieval_score": 0.9},
            )
        ]


class SurveyCountRetriever:
    def invoke(self, _query):
        return [
            Document(
                page_content=(
                    "Separately from response bias, the survey has gender details: "
                    "91% of respondents and 89% of non-respondents were male."
                ),
                metadata={"source": "survey.pdf", "page": 12, "retrieval_score": 0.95},
            ),
            Document(
                page_content=(
                    "Through a survey of leading AI/ML researchers, we explore technical experts' "
                    "attitudes about the governance of AI. We surveyed 524 AI/ML researchers in "
                    "September and October 2019."
                ),
                metadata={"source": "survey.pdf", "page": 2, "retrieval_score": 0.45},
            ),
        ]


class FranceRetriever:
    def invoke(self, _query):
        return [
            Document(
                page_content="The survey includes researchers from several countries including France and Germany.",
                metadata={"source": "survey.pdf", "page": 8, "retrieval_score": 0.8},
            )
        ]


def fake_llm():
    def respond(prompt):
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        if "strict query classifier" in text:
            return "IN_SCOPE"
        if "Generate retrieval query variants" in text:
            return "remote work policy\nwork from home allowance"
        if "enterprise RAG assistant" in text:
            return "Remote work is allowed two days per week [1]."
        if "strict hallucination checker" in text:
            return "VERDICT: PASS\nANSWER: Remote work is allowed two days per week [1]."
        return "IN_SCOPE"

    return RunnableLambda(respond)


def fake_pass_artifact_llm():
    def respond(prompt):
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        if "strict query classifier" in text:
            return "IN_SCOPE"
        if "Generate retrieval query variants" in text:
            return "remote work policy"
        if "enterprise RAG assistant" in text:
            return "Remote work is allowed two days per week [1]."
        if "strict hallucination checker" in text:
            return "VERDICT: PASS\nANSWER: PASS"
        return "IN_SCOPE"

    return RunnableLambda(respond)


def fake_incomplete_layer_llm():
    def respond(prompt):
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        if "strict query classifier" in text:
            return "IN_SCOPE"
        if "Generate retrieval query variants" in text:
            return "hourglass model layers"
        if "enterprise RAG assistant" in text:
            return "The Hourglass Model of Organizational AI Governance proposes AI governance components [1]."
        if "strict hallucination checker" in text:
            return "VERDICT: PASS\nANSWER: The Hourglass Model of Organizational AI Governance proposes AI governance components [1]."
        return "IN_SCOPE"

    return RunnableLambda(respond)


def fake_percentage_count_llm():
    def respond(prompt):
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        if "strict query classifier" in text:
            return "IN_SCOPE"
        if "Generate retrieval query variants" in text:
            return "surveyed AI/ML researchers count"
        if "enterprise RAG assistant" in text:
            return "91% of respondents and 89% of non-respondents were male [1]."
        if "strict hallucination checker" in text:
            return "VERDICT: PASS\nANSWER: 91% of respondents and 89% of non-respondents were male [1]."
        return "IN_SCOPE"

    return RunnableLambda(respond)


def fake_hallucinated_capital_llm():
    def respond(prompt):
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        if "strict query classifier" in text:
            return "IN_SCOPE"
        if "Generate retrieval query variants" in text:
            return "France capital"
        if "enterprise RAG assistant" in text:
            return "The capital of France is Paris."
        if "strict hallucination checker" in text:
            return "VERDICT: PASS\nANSWER: The capital of France is Paris."
        return "IN_SCOPE"

    return RunnableLambda(respond)


def test_agentic_pipeline_returns_citations_and_confidence(tmp_path):
    settings = Settings(
        vectorstore_dir=tmp_path / "idx",
        manifest_path=tmp_path / "idx" / "manifest.json",
        min_answer_confidence=0.2,
    )
    pipeline = AgenticRAGPipeline(
        retriever=FakeRetriever(),
        settings=settings,
        chat_model=fake_llm(),
    )

    result = pipeline.ask("What is the remote work policy?")

    assert result.verifier_passed is True
    assert result.confidence > 0.2
    assert result.citations[0].source == "policy.txt"
    assert "remote work" in result.answer.lower()


def test_pipeline_rejects_verifier_artifact_as_answer(tmp_path):
    settings = Settings(
        vectorstore_dir=tmp_path / "idx",
        manifest_path=tmp_path / "idx" / "manifest.json",
        min_answer_confidence=0.2,
    )
    pipeline = AgenticRAGPipeline(
        retriever=FakeRetriever(),
        settings=settings,
        chat_model=fake_pass_artifact_llm(),
    )

    result = pipeline.ask("What is the remote work policy?")

    assert result.verifier_passed is True
    assert result.answer == "The policy allows remote work two days per week. [1]"


def test_pipeline_prefers_extractive_answer_for_incomplete_list_answer(tmp_path):
    settings = Settings(
        vectorstore_dir=tmp_path / "idx",
        manifest_path=tmp_path / "idx" / "manifest.json",
        min_answer_confidence=0.2,
    )
    pipeline = AgenticRAGPipeline(
        retriever=LayerRetriever(),
        settings=settings,
        chat_model=fake_incomplete_layer_llm(),
    )

    result = pipeline.ask("What are the three layers in the hourglass model?")

    assert result.verifier_passed is True
    assert result.answer == "The three layers are environmental, organizational, and AI system. [1]"


def test_pipeline_composes_hourglass_metaphor_answer_from_evidence(tmp_path):
    settings = Settings(
        vectorstore_dir=tmp_path / "idx",
        manifest_path=tmp_path / "idx" / "manifest.json",
        min_answer_confidence=0.2,
    )
    pipeline = AgenticRAGPipeline(
        retriever=HourglassMetaphorRetriever(),
        settings=settings,
        chat_model=fake_incomplete_layer_llm(),
    )

    result = pipeline.ask("What does the hourglass metaphor represent in AI governance?")

    assert result.verifier_passed is True
    assert result.answer == (
        "The hourglass metaphor represents governance requirements flowing from the "
        "environmental layer to AI systems through the mediating organizational layer. [1]"
    )


def test_pipeline_composes_head_of_ai_answer_from_evidence(tmp_path):
    settings = Settings(
        vectorstore_dir=tmp_path / "idx",
        manifest_path=tmp_path / "idx" / "manifest.json",
        min_answer_confidence=0.2,
    )
    pipeline = AgenticRAGPipeline(
        retriever=HeadOfAIRetriever(),
        settings=settings,
        chat_model=fake_incomplete_layer_llm(),
    )

    result = pipeline.ask("What role does the paper assign to a Head of AI?")

    assert result.verifier_passed is True
    assert "oversees AI system development and operations" in result.answer


def test_pipeline_composes_survey_findings_from_evidence(tmp_path):
    settings = Settings(
        vectorstore_dir=tmp_path / "idx",
        manifest_path=tmp_path / "idx" / "manifest.json",
        min_answer_confidence=0.2,
    )
    pipeline = AgenticRAGPipeline(
        retriever=SurveyFindingsRetriever(),
        settings=settings,
        chat_model=fake_incomplete_layer_llm(),
    )

    trust = pipeline.ask(
        "Which institutions did surveyed researchers trust more for shaping AI in the public interest?"
    )
    weapons = pipeline.ask(
        "How did respondents feel about AI/ML researchers working on lethal autonomous weapons?"
    )
    safety = pipeline.ask("What did the survey find about AI safety research prioritization?")
    prepub = pipeline.ask("What pre-publication practice did many respondents support?")

    assert "international organizations and scientific organizations" in trust.answer
    assert "overwhelmingly opposed" in weapons.answer
    assert "AI safety research should be prioritized" in safety.answer
    assert "pre-publication review" in prepub.answer


def test_pipeline_normalizes_pdf_artifacts_for_trust_answer(tmp_path):
    settings = Settings(
        vectorstore_dir=tmp_path / "idx",
        manifest_path=tmp_path / "idx" / "manifest.json",
        min_answer_confidence=0.2,
    )
    pipeline = AgenticRAGPipeline(
        retriever=PdfArtifactTrustRetriever(),
        settings=settings,
        chat_model=fake_incomplete_layer_llm(),
    )

    result = pipeline.ask(
        "Which institutions did surveyed researchers trust more for shaping AI in the public interest?"
    )

    assert result.verifier_passed is True
    assert "international organizations and scientific organizations" in result.answer


def test_pipeline_prefers_survey_count_over_percentage_for_how_many(tmp_path):
    settings = Settings(
        vectorstore_dir=tmp_path / "idx",
        manifest_path=tmp_path / "idx" / "manifest.json",
        min_answer_confidence=0.2,
    )
    pipeline = AgenticRAGPipeline(
        retriever=SurveyCountRetriever(),
        settings=settings,
        chat_model=fake_percentage_count_llm(),
    )

    result = pipeline.ask("How many AI/ML researchers were surveyed?")

    assert result.verifier_passed is True
    assert result.answer == "The survey included 524 AI/ML researchers. [2]"


def test_pipeline_rejects_generated_answer_with_terms_absent_from_context(tmp_path):
    settings = Settings(
        vectorstore_dir=tmp_path / "idx",
        manifest_path=tmp_path / "idx" / "manifest.json",
        min_answer_confidence=0.2,
    )
    pipeline = AgenticRAGPipeline(
        retriever=FranceRetriever(),
        settings=settings,
        chat_model=fake_hallucinated_capital_llm(),
    )

    result = pipeline.ask("What is the capital of France?")

    assert result.verifier_passed is False
    assert result.answer == "Not found in document."
