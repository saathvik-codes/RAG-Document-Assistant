from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from rag_assistant.agents import REFUSAL_ANSWERS, RAGResult


@dataclass
class EvaluationSummary:
    total: int
    answered: int
    not_found: int
    verifier_pass_rate: float
    average_confidence: float


def summarize_results(results: Iterable[RAGResult]) -> EvaluationSummary:
    result_list = list(results)
    total = len(result_list)
    answered = sum(1 for item in result_list if item.answer.lower() not in REFUSAL_ANSWERS)
    not_found = total - answered
    verifier_passed = sum(1 for item in result_list if item.verifier_passed)
    confidence_total = sum(item.confidence for item in result_list)
    return EvaluationSummary(
        total=total,
        answered=answered,
        not_found=not_found,
        verifier_pass_rate=round(verifier_passed / total, 3) if total else 0.0,
        average_confidence=round(confidence_total / total, 3) if total else 0.0,
    )
