from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, List

from langchain_core.documents import Document


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]{2,}")


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


@dataclass
class RetrievedDocument:
    document: Document
    dense_rank: int | None
    keyword_score: float
    combined_score: float


class HybridRetriever:
    """Dense FAISS retrieval plus lightweight local keyword scoring."""

    def __init__(self, vectorstore: Any, chunks: List[Document], settings: Any):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.settings = settings
        self._doc_tokens = [tokenize(doc.page_content) for doc in chunks]
        self._doc_freq = self._build_doc_freq(self._doc_tokens)
        self._avg_doc_len = (
            sum(len(tokens) for tokens in self._doc_tokens) / len(self._doc_tokens)
            if self._doc_tokens
            else 0
        )

    @staticmethod
    def _build_doc_freq(token_lists: Iterable[List[str]]) -> dict[str, int]:
        freq: dict[str, int] = defaultdict(int)
        for tokens in token_lists:
            for token in set(tokens):
                freq[token] += 1
        return dict(freq)

    def _doc_key(self, doc: Document) -> tuple[str, str, str]:
        return (
            str(doc.metadata.get("source", "Unknown")),
            str(doc.metadata.get("page", "N/A")),
            doc.page_content[:220],
        )

    def _keyword_scores(self, query: str) -> dict[int, float]:
        query_terms = tokenize(query)
        if not query_terms or not self.chunks:
            return {}

        scores: dict[int, float] = {}
        total_docs = len(self.chunks)
        k1 = 1.5
        b = 0.75

        for idx, tokens in enumerate(self._doc_tokens):
            if not tokens:
                continue
            token_counts = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for term in query_terms:
                term_freq = token_counts.get(term, 0)
                if term_freq == 0:
                    continue
                doc_freq = self._doc_freq.get(term, 0)
                idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                denom = term_freq + k1 * (1 - b + b * doc_len / max(self._avg_doc_len, 1))
                score += idf * (term_freq * (k1 + 1)) / denom
            if score > 0:
                scores[idx] = score
        return scores

    def _dense_docs(self, query: str) -> List[Document]:
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.settings.retrieval_k,
                "fetch_k": max(self.settings.retrieval_fetch_k, self.settings.retrieval_k),
                "lambda_mult": self.settings.retrieval_lambda_mult,
            },
        )
        return retriever.invoke(query)

    def _source_affinity(self, query: str, doc: Document) -> float:
        query_l = query.lower()
        source_l = str(doc.metadata.get("source", "")).lower()

        survey_terms = {
            "survey",
            "surveyed",
            "respondent",
            "respondents",
            "researchers",
            "lethal",
            "autonomous weapons",
            "pre-publication",
            "publication",
            "safety research",
            "trust",
            "institutions",
        }
        governance_terms = {
            "hourglass",
            "organizational layer",
            "environmental layer",
            "ai system layer",
            "head of ai",
            "strategic alignment",
            "value alignment",
        }
        nvidia_terms = {
            "nvidia",
            "revenue",
            "data center",
            "compute",
            "networking",
            "graphics",
            "fiscal",
            "10-k",
            "sec",
        }

        if any(term in query_l for term in survey_terms) and (
            "survey" in source_l or "researcher" in source_l or "ethics" in source_l
        ):
            return 0.35
        if any(term in query_l for term in governance_terms) and (
            "governance" in source_l or "hourglass" in source_l
        ):
            return 0.25
        if any(term in query_l for term in nvidia_terms) and "nvidia" in source_l:
            return 0.35
        return 0.0

    def invoke(self, query: str) -> List[Document]:
        dense_docs = self._dense_docs(query)
        keyword_scores = self._keyword_scores(query)
        dense_by_key = {self._doc_key(doc): rank for rank, doc in enumerate(dense_docs, start=1)}
        chunk_by_key = {self._doc_key(doc): (idx, doc) for idx, doc in enumerate(self.chunks)}

        candidate_keys = set(dense_by_key.keys())
        top_keyword = sorted(keyword_scores.items(), key=lambda item: item[1], reverse=True)[
            : self.settings.keyword_retrieval_k
        ]
        for idx, _score in top_keyword:
            candidate_keys.add(self._doc_key(self.chunks[idx]))

        max_keyword = max(keyword_scores.values(), default=1.0)
        results: List[RetrievedDocument] = []
        for key in candidate_keys:
            idx, doc = chunk_by_key.get(key, (-1, None))
            if doc is None:
                doc = next(item for item in dense_docs if self._doc_key(item) == key)
                idx = -1
            dense_rank = dense_by_key.get(key)
            dense_score = 1.0 / dense_rank if dense_rank else 0.0
            keyword_score = (keyword_scores.get(idx, 0.0) / max_keyword) if idx >= 0 else 0.0
            combined = 0.4 * dense_score + 0.5 * keyword_score + self._source_affinity(query, doc)
            doc.metadata["retrieval_score"] = round(combined, 4)
            doc.metadata["keyword_score"] = round(keyword_score, 4)
            doc.metadata["dense_rank"] = dense_rank
            results.append(
                RetrievedDocument(
                    document=doc,
                    dense_rank=dense_rank,
                    keyword_score=keyword_score,
                    combined_score=combined,
                )
            )

        results.sort(key=lambda item: item.combined_score, reverse=True)
        return [item.document for item in results[: self.settings.max_context_docs]]
