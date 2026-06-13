from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests


def _contains_all(answer: str, needles: list[str]) -> bool:
    answer_l = answer.lower()
    return all(needle.lower() in answer_l for needle in needles)


def _case_passed(case: dict[str, Any], result: dict[str, Any]) -> tuple[bool, str]:
    answer = str(result.get("answer", ""))
    answered = answer.strip().lower() != "not found in document."
    if bool(case["expect_answered"]) != answered:
        expected = "answered" if case["expect_answered"] else "not found"
        return False, f"expected {expected}, got: {answer}"

    if not answered:
        return True, "refused unsupported question"

    must_include = case.get("must_include", [])
    if must_include and not _contains_all(answer, must_include):
        return False, f"missing required terms {must_include}; got: {answer}"

    must_include_any = case.get("must_include_any", [])
    if must_include_any and not any(_contains_all(answer, group) for group in must_include_any):
        return False, f"missing one required term group {must_include_any}; got: {answer}"

    if not result.get("verifier_passed", False):
        return False, "answer was produced but verifier_passed is false"

    if float(result.get("confidence", 0.0)) <= 0:
        return False, "answer was produced but confidence is zero"

    return True, answer


def run_eval(base_url: str, cases_path: Path, timeout: int) -> int:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    print(f"Running {len(cases)} eval cases against {base_url}")
    for case in cases:
        response = requests.post(
            f"{base_url.rstrip('/')}/ask",
            json={"query": case["query"]},
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        passed, detail = _case_passed(case, result)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {case['id']}: {detail}")
        if not passed:
            failures.append(case["id"])

    if failures:
        print(f"\nFailed cases: {', '.join(failures)}")
        return 1

    print("\nAll eval cases passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG quality evals against the FastAPI backend.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--cases", default="eval/eval_cases.json")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    return run_eval(args.base_url, Path(args.cases), args.timeout)


if __name__ == "__main__":
    sys.exit(main())
