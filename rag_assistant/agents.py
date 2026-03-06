from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag_assistant.config import Settings


CLASSIFIER_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict query classifier for enterprise document QA.
Classify the user query into one label:
- IN_SCOPE: the query could be answered from uploaded documents.
- OUT_OF_SCOPE: casual chat, opinions, or requests not tied to uploaded documents.

Return only one token: IN_SCOPE or OUT_OF_SCOPE.
Query: {query}
"""
)


REWRITER_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the user query to improve document retrieval quality.
Keep meaning unchanged. Keep it short and factual.
If rewrite is unnecessary, return the original query exactly.
Return only the rewritten query text on one line.
Do not add explanations, prefixes, markdown, quotes, or code fences.

Original query: {query}
"""
)


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an enterprise RAG assistant.
Answer only from the context below.
If the answer is not explicitly in context, output exactly: Not found in document.
Do not use outside knowledge.

Question: {question}

Context:
{context}

Output style:
- First line: concise answer text
- If answer exists, include short inline citation markers like [1], [2]
"""
)


VERIFIER_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict hallucination checker.
Check whether every factual claim in the draft answer is supported by the provided context.
If unsupported or partially unsupported, fail it.

Question: {question}
Context:
{context}
Draft answer:
{draft_answer}

Return exactly two lines:
VERDICT: PASS or FAIL
ANSWER: <final answer text>

Rules:
- If FAIL, ANSWER must be exactly: Not found in document.
- If PASS, ANSWER should preserve the draft answer intent with no new facts.
"""
)


@dataclass
class RAGResult:
    answer: str
    sources: List[str]
    rewritten_query: str
    classifier_label: str
    verifier_passed: bool


class AgenticRAGPipeline:
    def __init__(self, retriever, settings: Settings):
        self.retriever = retriever
        self.settings = settings
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0,
        )
        self.parser = StrOutputParser()

    def _format_context(self, docs: List[Document]) -> str:
        blocks: List[str] = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            blocks.append(f"[{idx}] Source: {source} | Page: {page}\n{doc.page_content}")
        return "\n\n".join(blocks)

    def _format_sources(self, docs: List[Document]) -> List[str]:
        seen = set()
        sources: List[str] = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            key = (source, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append(f"{source} (Page {page})")
        return sources

    def _classify_query(self, query: str) -> str:
        raw = (CLASSIFIER_PROMPT | self.llm | self.parser).invoke({"query": query}).strip()
        label = raw.split()[0].upper() if raw else "IN_SCOPE"
        if "OUT_OF_SCOPE" in label:
            return "OUT_OF_SCOPE"
        return "IN_SCOPE"

    def _sanitize_rewritten_query(self, original_query: str, rewritten_query: str) -> str:
        candidate = rewritten_query.strip()
        if not candidate:
            return original_query

        candidate = candidate.replace("```", "").strip()
        candidate = re.sub(r"^rewritten query\s*[:\-]\s*", "", candidate, flags=re.IGNORECASE)

        quoted_matches = re.findall(r'"([^"\n]{3,300})"', candidate)
        if quoted_matches:
            candidate = quoted_matches[-1].strip()
        else:
            lines = [line.strip("` ").strip() for line in candidate.splitlines() if line.strip()]
            filtered_lines = [
                line
                for line in lines
                if not re.search(
                    r"(original query|rewritten query|no rewriting|identical|verifier|passed|query would be)",
                    line,
                    flags=re.IGNORECASE,
                )
            ]
            if filtered_lines:
                candidate = filtered_lines[0]
            elif lines:
                candidate = lines[0]
            else:
                candidate = original_query

        candidate = candidate.strip("`'\" ").strip()
        if (
            not candidate
            or len(candidate) > 300
            or re.search(
                r"(original query|rewritten query|verifier|passed|^the original query is)",
                candidate,
                flags=re.IGNORECASE,
            )
        ):
            return original_query
        return candidate

    def _rewrite_query(self, query: str) -> str:
        raw_rewrite = (REWRITER_PROMPT | self.llm | self.parser).invoke({"query": query}).strip()
        return self._sanitize_rewritten_query(query, raw_rewrite)

    def _doc_key(self, doc: Document) -> tuple[str, str, str, str]:
        source = str(doc.metadata.get("source", "Unknown"))
        page = str(doc.metadata.get("page", "N/A"))
        section = str(doc.metadata.get("section", ""))
        content_head = doc.page_content[:160]
        return (source, page, section, content_head)

    def _retrieve_documents(self, query: str, rewritten_query: str) -> List[Document]:
        combined_docs: List[Document] = []
        seen = set()
        retrieval_queries = [rewritten_query]
        if rewritten_query.strip().lower() != query.strip().lower():
            retrieval_queries.append(query)

        for retrieval_query in retrieval_queries:
            docs = self.retriever.invoke(retrieval_query)
            for doc in docs:
                key = self._doc_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                combined_docs.append(doc)
                if len(combined_docs) >= self.settings.max_context_docs:
                    return combined_docs

        return combined_docs[: self.settings.max_context_docs]

    def _generate_answer(self, question: str, context: str) -> str:
        answer = (ANSWER_PROMPT | self.llm | self.parser).invoke(
            {"question": question, "context": context}
        )
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        return lines[0] if lines else "Not found in document."

    def _verify_answer(self, question: str, context: str, draft_answer: str) -> tuple[bool, str]:
        raw = (VERIFIER_PROMPT | self.llm | self.parser).invoke(
            {"question": question, "context": context, "draft_answer": draft_answer}
        )
        verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", raw, flags=re.IGNORECASE)
        if verdict_match is None:
            verdict_match = re.search(r"\b(PASS|FAIL)\b", raw, flags=re.IGNORECASE)
        answer_match = re.search(r"ANSWER:\s*(.*)", raw, flags=re.IGNORECASE | re.DOTALL)
        verdict = verdict_match.group(1).upper() if verdict_match else "FAIL"
        final_answer = answer_match.group(1).strip() if answer_match else draft_answer.strip()
        final_lines = [line.strip() for line in final_answer.splitlines() if line.strip()]
        final_answer = final_lines[0] if final_lines else "Not found in document."
        if verdict != "PASS":
            return False, "Not found in document."
        if final_answer.lower() == "not found in document.":
            return False, "Not found in document."
        return True, final_answer

    def ask(self, query: str) -> RAGResult:
        classifier_label = self._classify_query(query)
        if classifier_label == "OUT_OF_SCOPE":
            return RAGResult(
                answer="Please ask a question related to the uploaded documents.",
                sources=[],
                rewritten_query=query,
                classifier_label=classifier_label,
                verifier_passed=False,
            )

        rewritten_query = self._rewrite_query(query)
        retrieved_docs = self._retrieve_documents(query, rewritten_query)

        if not retrieved_docs:
            return RAGResult(
                answer="Not found in document.",
                sources=[],
                rewritten_query=rewritten_query,
                classifier_label=classifier_label,
                verifier_passed=False,
            )

        context = self._format_context(retrieved_docs)
        draft_answer = self._generate_answer(query, context)
        verifier_passed, final_answer = self._verify_answer(query, context, draft_answer)
        sources = self._format_sources(retrieved_docs)

        return RAGResult(
            answer=final_answer,
            sources=sources,
            rewritten_query=rewritten_query,
            classifier_label=classifier_label,
            verifier_passed=verifier_passed,
        )
