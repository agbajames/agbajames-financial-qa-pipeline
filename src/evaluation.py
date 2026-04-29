"""
Evaluation module for Financial QA pipeline.

Submission-ready scope for the PwC take-home:
    - Exact Match
    - Token-level F1
    - ROUGE-L
    - Numerical accuracy
    - Abstention metrics
    - Lightweight operational metrics (latency and token usage)

This version intentionally removes:
    - BERTScore
    - Calibration / Brier score
    - LLM-as-judge

The assignment prioritises clarity, reproducibility, and pragmatic judgment
over complexity, so this module keeps the strongest task-relevant metrics.
"""

from __future__ import annotations

import csv
import json
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


_NUMBER_TOKEN_RE = re.compile(
    r"(?<!\w)-?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?!\w)"
)
_DASH_TRANSLATION = str.maketrans({"−": "-", "–": "-", "—": "-"})


def normalise_text(text: str) -> str:
    """Lowercase text, preserve numeric values, and collapse whitespace."""
    text = str(text).lower().translate(_DASH_TRANSLATION).strip()
    text = re.sub(r"(?<=\d)\s*%", " percent ", text)
    text = text.replace("$", " ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    protected_numbers: List[str] = []

    def _protect(match: re.Match[str]) -> str:
        protected_numbers.append(match.group(0))
        return f" __num_{len(protected_numbers) - 1}__ "

    text = _NUMBER_TOKEN_RE.sub(_protect, text)

    punctuation_to_space = {ch: " " for ch in string.punctuation if ch != "_"}
    text = text.translate(str.maketrans(punctuation_to_space))
    text = re.sub(r"\s+", " ", text).strip()

    for i, value in enumerate(protected_numbers):
        text = text.replace(f"__num_{i}__", value)

    return text


def tokenise(text: str) -> List[str]:
    """Whitespace tokeniser over normalised text."""
    return normalise_text(text).split()


def exact_match(predicted: str, reference: str) -> float:
    """Normalised exact match. Returns 1.0 or 0.0."""
    return 1.0 if normalise_text(predicted) == normalise_text(reference) else 0.0


def token_f1(predicted: str, reference: str) -> Dict[str, float]:
    """Compute token-level precision, recall, and F1."""
    pred_tokens = tokenise(predicted)
    ref_tokens = tokenise(reference)

    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = set(pred_tokens) & set(ref_tokens)
    num_common = sum(min(pred_tokens.count(w), ref_tokens.count(w)) for w in common)

    if num_common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute the length of the longest common subsequence."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def rouge_l(predicted: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-L precision, recall, and F1."""
    pred_tokens = tokenise(predicted)
    ref_tokens = tokenise(reference)

    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


_SCALE_MULTIPLIERS = {
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}
_NUMBER_VALUE_RE = r"-?(?:\d+(?:\.\d+)?|\.\d+)"


def _normalise_numeric_text(text: str) -> str:
    """Normalise numeric text for extraction."""
    text = str(text).translate(_DASH_TRANSLATION)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    def _replace_accounting_negative(match: re.Match[str]) -> str:
        number = match.group("number")
        suffix = match.group("suffix") or ""
        return f"-{number} {suffix}".strip()

    return re.sub(
        rf"\(\s*[$€£]?\s*(?P<number>{_NUMBER_VALUE_RE})\s*(?P<suffix>%|percent|thousand|million|billion|trillion)?\s*\)",
        _replace_accounting_negative,
        text,
        flags=re.IGNORECASE,
    )


def _span_overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    """Return True when a span overlaps any previously occupied span."""
    start, end = span
    return any(not (end <= left or start >= right) for left, right in occupied)


def extract_numbers(text: str) -> List[float]:
    """
    Extract numeric values from text.

    Handles:
        - $1.35 billion
        - 45.2%
        - 45.2 percent
        - 1,234,567
        - -5.2
        - ($123 million)
    """
    text = _normalise_numeric_text(text)
    numbers: List[float] = []
    occupied: List[tuple[int, int]] = []

    scaled_pattern = re.compile(
        rf"(?<!\w)[$€£]?(?P<value>{_NUMBER_VALUE_RE})\s*(?P<scale>thousand|million|billion|trillion)(?!\w)",
        re.IGNORECASE,
    )
    percent_pattern = re.compile(
        rf"(?<!\w)(?P<value>{_NUMBER_VALUE_RE})\s*(?:%|percent)(?!\w)",
        re.IGNORECASE,
    )
    standalone_pattern = re.compile(
        rf"(?<!\w)[$€£]?(?P<value>{_NUMBER_VALUE_RE})(?!\w)",
        re.IGNORECASE,
    )

    for match in scaled_pattern.finditer(text):
        value = float(match.group("value"))
        scale = match.group("scale").lower()
        numbers.append(value * _SCALE_MULTIPLIERS[scale])
        occupied.append(match.span())

    for match in percent_pattern.finditer(text):
        if _span_overlaps(match.span(), occupied):
            continue
        numbers.append(float(match.group("value")))
        occupied.append(match.span())

    for match in standalone_pattern.finditer(text):
        if _span_overlaps(match.span(), occupied):
            continue
        numbers.append(float(match.group("value")))

    return numbers


def numerical_accuracy(predicted: str, reference: str) -> Dict[str, Any]:
    """Compare numeric values in predicted vs reference answers."""
    pred_nums = extract_numbers(predicted)
    ref_nums = extract_numbers(reference)

    if not ref_nums:
        return {
            "has_numbers": False,
            "matched": 0,
            "total_ref": 0,
            "mean_relative_error": None,
            "accuracy": None,
            "details": [],
        }

    details: List[Dict[str, Any]] = []
    matched = 0
    rel_errors: List[float] = []
    tolerance = 0.05
    used_pred: set[int] = set()

    for ref_val in ref_nums:
        best_error = float("inf")
        best_pred = None
        best_idx = None

        for i, pred_val in enumerate(pred_nums):
            if i in used_pred:
                continue
            error = abs(pred_val) if ref_val == 0 else abs(pred_val - ref_val) / abs(ref_val)
            if error < best_error:
                best_error = error
                best_pred = pred_val
                best_idx = i

        is_match = best_error <= tolerance
        if is_match and best_idx is not None:
            matched += 1
            used_pred.add(best_idx)

        rel_errors.append(best_error if best_pred is not None else 1.0)
        details.append(
            {
                "reference": ref_val,
                "predicted": best_pred,
                "relative_error": round(best_error, 4) if best_pred is not None else None,
                "match": is_match,
            }
        )

    mean_rel_error = float(np.mean(rel_errors)) if rel_errors else None

    return {
        "has_numbers": True,
        "matched": matched,
        "total_ref": len(ref_nums),
        "mean_relative_error": round(mean_rel_error, 4) if mean_rel_error is not None else None,
        "accuracy": round(matched / len(ref_nums), 4),
        "details": details,
    }


def abstention_metrics(
    records: List[Dict[str, Any]],
    quality_key: str = "token_f1",
    poor_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Compute abstention-related metrics.

    Returns coverage, accuracy@answered, should-have-abstained rate,
    and a simple per-answer-type breakdown.
    """
    valid_records = [r for r in records if not r.get("error")]
    total = len(valid_records)
    if total == 0:
        return {}

    answered = [r for r in valid_records if not r.get("abstain", False)]
    abstained = [r for r in valid_records if r.get("abstain", False)]

    coverage = len(answered) / total
    answered_scores = [r.get(quality_key, 0.0) for r in answered]
    accuracy_at_answered = float(np.mean(answered_scores)) if answered_scores else 0.0

    should_have_abstained = [
        r for r in answered if r.get(quality_key, 0.0) < poor_threshold
    ]
    should_have_abstained_rate = (
        len(should_have_abstained) / len(answered) if answered else 0.0
    )

    by_type = defaultdict(lambda: {"total": 0, "answered": 0, "abstained": 0})
    for record in valid_records:
        answer_type = record.get("answer_type", "unknown")
        by_type[answer_type]["total"] += 1
        if record.get("abstain", False):
            by_type[answer_type]["abstained"] += 1
        else:
            by_type[answer_type]["answered"] += 1

    return {
        "total_samples": total,
        "answered": len(answered),
        "abstained": len(abstained),
        "coverage": round(coverage, 4),
        "accuracy_at_answered": round(accuracy_at_answered, 4),
        "should_have_abstained": len(should_have_abstained),
        "should_have_abstained_rate": round(should_have_abstained_rate, 4),
        "by_answer_type": dict(by_type),
    }


def evaluate_single(predicted: str, reference: str) -> Dict[str, Any]:
    """Compute all per-sample metrics for a single prediction."""
    em = exact_match(predicted, reference)
    tf1 = token_f1(predicted, reference)
    rl = rouge_l(predicted, reference)
    num = numerical_accuracy(predicted, reference)

    return {
        "exact_match": em,
        "token_f1": tf1["f1"],
        "token_precision": tf1["precision"],
        "token_recall": tf1["recall"],
        "rouge_l": rl["f1"],
        "rouge_l_precision": rl["precision"],
        "rouge_l_recall": rl["recall"],
        "numerical_accuracy": num["accuracy"],
        "numerical_has_numbers": num["has_numbers"],
        "numerical_mean_rel_error": num["mean_relative_error"],
        "numerical_details": num["details"],
    }


def evaluate_predictions(
    records: List[Dict[str, Any]],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate prediction records for one model.

    Expected fields per record:
        - predicted_answer
        - reference_answer
        - answer_type
        - abstain
        - confidence
    """
    if verbose:
        print(f"Evaluating {len(records)} predictions...")

    for record in records:
        if record.get("abstain", False) or record.get("error"):
            record.update(
                {
                    "exact_match": 0.0,
                    "token_f1": 0.0,
                    "token_precision": 0.0,
                    "token_recall": 0.0,
                    "rouge_l": 0.0,
                    "rouge_l_precision": 0.0,
                    "rouge_l_recall": 0.0,
                    "numerical_accuracy": None,
                    "numerical_has_numbers": False,
                    "numerical_mean_rel_error": None,
                    "numerical_details": [],
                }
            )
            continue

        record.update(
            evaluate_single(
                predicted=record["predicted_answer"],
                reference=record["reference_answer"],
            )
        )

    answered = [r for r in records if not r.get("abstain", False) and not r.get("error")]
    abstained = [r for r in records if r.get("abstain", False) and not r.get("error")]
    errored = [r for r in records if r.get("error")]

    def safe_mean(values: List[Any]) -> float | None:
        filtered = [v for v in values if v is not None]
        return round(float(np.mean(filtered)), 4) if filtered else None

    abstention = abstention_metrics(records, quality_key="token_f1")

    summary = {
        "n_total": len(records),
        "n_answered": len(answered),
        "n_abstained": len(abstained),
        "n_errors": len(errored),
        "coverage": abstention.get("coverage"),
        "accuracy_at_answered": abstention.get("accuracy_at_answered"),
        "should_have_abstained_rate": abstention.get("should_have_abstained_rate"),
        "mean_exact_match": safe_mean([r["exact_match"] for r in answered]),
        "mean_token_f1": safe_mean([r["token_f1"] for r in answered]),
        "mean_rouge_l": safe_mean([r["rouge_l"] for r in answered]),
        "mean_numerical_accuracy": safe_mean(
            [
                r["numerical_accuracy"]
                for r in answered
                if r.get("numerical_has_numbers")
            ]
        ),
        "mean_numerical_rel_error": safe_mean(
            [
                r["numerical_mean_rel_error"]
                for r in answered
                if r.get("numerical_has_numbers")
            ]
        ),
        "mean_latency_ms": safe_mean([r.get("latency_ms") for r in records]),
        "total_input_tokens": sum(r.get("input_tokens", 0) or 0 for r in records),
        "total_output_tokens": sum(r.get("output_tokens", 0) or 0 for r in records),
        "total_tokens": sum(r.get("total_tokens", 0) or 0 for r in records),
        "mean_confidence": safe_mean([r.get("confidence") for r in answered]),
    }

    return {
        "records": records,
        "summary": summary,
        "abstention": abstention,
    }


def save_evaluation(
    eval_result: Dict[str, Any],
    model_name: str,
    output_dir: str | Path = "outputs",
) -> Dict[str, Path]:
    """Save evaluation outputs as a summary JSON and flat per-sample CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.replace("/", "_").replace(".", "-")

    summary_path = output_dir / f"eval_summary_{safe_name}.json"
    summary_data = {
        "model": model_name,
        "summary": eval_result["summary"],
        "abstention": eval_result["abstention"],
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary_data, file, indent=2, default=str)

    csv_path = output_dir / f"eval_details_{safe_name}.csv"
    csv_fields = [
        "sample_id",
        "ticker",
        "filing",
        "answer_type",
        "model",
        "abstain",
        "confidence",
        "abstain_reason_code",
        "exact_match",
        "token_f1",
        "rouge_l",
        "numerical_accuracy",
        "numerical_mean_rel_error",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "predicted_answer",
        "reference_answer",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for record in eval_result["records"]:
            writer.writerow(record)

    print(f"Saved evaluation for {model_name}:")
    print(f"  Summary: {summary_path}")
    print(f"  Details: {csv_path}")

    return {"summary": summary_path, "details": csv_path}


if __name__ == "__main__":
    print("=== Evaluation Module Self-Test ===\n")

    predicted = (
        "NVIDIA recorded an acquisition termination cost of "
        "$1.35 billion in fiscal year 2023."
    )
    reference = (
        "NVIDIA recorded an acquisition termination cost of "
        "$1.35 billion in fiscal year 2023."
    )

    print(f"Exact match: {exact_match(predicted, reference)}")
    print(f"Token F1: {token_f1(predicted, reference)}")
    print(f"ROUGE-L: {rouge_l(predicted, reference)}")
    print(f"Numerical: {numerical_accuracy(predicted, reference)}")
