# file: src/pipeline.py
"""
Pipeline runner for the Financial QA take-home.

Runs one or more OpenAI chat models over the sampled 10-K questions,
enforces structured outputs, records latency/token usage, and saves
predictions as JSONL.

Usage:
    python -m src.pipeline --config config.yaml
    python src/pipeline.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError("pip install openai  (requires openai>=1.0)") from exc

from src.prompt_and_schema import build_chat_payload, postprocess_response

DEFAULT_MODELS = ["gpt-4.1", "gpt-4.1-mini"]
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_CONFIG_PATH = "config.yaml"


def _safe_model_name(model_name: str) -> str:
    """Convert a model name into a filesystem-safe suffix."""
    return model_name.replace("/", "_").replace(".", "-")


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load YAML config and fill simple defaults."""
    config_path = Path(config_path)

    if not config_path.exists():
        return {
            "dataset": {
                "csv_path": "Financial-QA-10k.csv",
                "sample_size": 50,
                "random_seed": 42,
            },
            "models": [{"name": name} for name in DEFAULT_MODELS],
            "pipeline": {
                "temperature": DEFAULT_TEMPERATURE,
                "max_retries": DEFAULT_MAX_RETRIES,
                "retry_delay_seconds": DEFAULT_RETRY_DELAY,
            },
            "output": {
                "predictions_dir": "outputs",
                "evaluation_dir": "outputs",
                "slides_dir": "slides",
            },
        }

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config.setdefault("dataset", {})
    config["dataset"].setdefault("csv_path", "Financial-QA-10k.csv")
    config["dataset"].setdefault("sample_size", 50)
    config["dataset"].setdefault("random_seed", 42)

    config.setdefault("models", [{"name": name} for name in DEFAULT_MODELS])

    config.setdefault("pipeline", {})
    config["pipeline"].setdefault("temperature", DEFAULT_TEMPERATURE)
    config["pipeline"].setdefault("max_retries", DEFAULT_MAX_RETRIES)
    config["pipeline"].setdefault("retry_delay_seconds", DEFAULT_RETRY_DELAY)

    config.setdefault("output", {})
    config["output"].setdefault("predictions_dir", "outputs")
    config["output"].setdefault("evaluation_dir", "outputs")
    config["output"].setdefault("slides_dir", "slides")

    return config


def extract_model_names(config: Dict[str, Any]) -> List[str]:
    """Extract model names from config, supporting strings or dict entries."""
    models = config.get("models", [])
    names: List[str] = []

    for entry in models:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict) and entry.get("name"):
            names.append(str(entry["name"]))

    return names or list(DEFAULT_MODELS)


def call_model(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> Dict[str, Any]:
    """
    Call one model for one question-context pair.

    Returns parsed structured output plus latency, token usage, and errors.
    """
    payload = build_chat_payload(
        model=model,
        question=question,
        context=context,
        temperature=temperature,
    )

    error_msg = None

    for attempt in range(max_retries + 1):
        try:
            started = time.perf_counter()
            response = client.chat.completions.create(**payload)
            latency_ms = (time.perf_counter() - started) * 1000

            raw_content = response.choices[0].message.content or ""
            parsed = json.loads(raw_content)
            parsed = postprocess_response(parsed, context=context, strict=False)

            usage = getattr(response, "usage", None)

            return {
                "model_response": parsed,
                "latency_ms": round(latency_ms, 1),
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "error": None,
                "retries": attempt,
            }

        except json.JSONDecodeError as exc:
            error_msg = f"JSON parse error: {exc}"
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"

        if attempt < max_retries:
            time.sleep(retry_delay * (attempt + 1))

    return {
        "model_response": None,
        "latency_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "error": error_msg,
        "retries": max_retries,
    }


def build_prediction_record(
    sample_row: Dict[str, Any],
    model_name: str,
    call_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine sample metadata and model output into one prediction record."""
    response_obj = call_result["model_response"] or {}

    return {
        "sample_id": sample_row["sample_id"],
        "question": sample_row["question"],
        "reference_answer": sample_row["answer"],
        "context": sample_row["context"],
        "ticker": sample_row["ticker"],
        "filing": sample_row["filing"],
        "answer_type": sample_row["answer_type"],
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_answer": response_obj.get("answer", ""),
        "confidence": response_obj.get("confidence"),
        "abstain": response_obj.get("abstain", False),
        "abstain_reason": response_obj.get("abstain_reason"),
        "abstain_reason_code": response_obj.get("abstain_reason_code"),
        "confidence_note": response_obj.get("confidence_note"),
        "supporting_evidence": response_obj.get("supporting_evidence", []),
        "latency_ms": call_result["latency_ms"],
        "input_tokens": call_result["input_tokens"],
        "output_tokens": call_result["output_tokens"],
        "total_tokens": call_result["total_tokens"],
        "error": call_result["error"],
        "retries": call_result["retries"],
        "contract_violations": response_obj.get("_contract_violations", []),
    }


def run_model_pipeline(
    client: OpenAI,
    model: str,
    samples: pd.DataFrame,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run one model across all samples."""
    records: List[Dict[str, Any]] = []
    total = len(samples)

    for index, (_, row) in enumerate(samples.iterrows(), start=1):
        if verbose:
            print(
                f"  [{index}/{total}] sample_id={row['sample_id']} "
                f"ticker={row['ticker']} type={row['answer_type']}",
                end=" ... ",
                flush=True,
            )

        result = call_model(
            client=client,
            model=model,
            question=row["question"],
            context=row["context"],
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        record = build_prediction_record(
            sample_row=row.to_dict(),
            model_name=model,
            call_result=result,
        )
        records.append(record)

        if verbose:
            if record["error"]:
                status = f"ERROR ({record['error']})"
            elif record["abstain"]:
                status = "ABSTAIN"
            else:
                status = f"conf={record['confidence']}"
            latency = (
                f"{record['latency_ms']}ms"
                if record["latency_ms"] is not None
                else "latency=n/a"
            )
            print(f"{status} ({latency})")

    return records


def save_predictions(
    records: List[Dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save prediction records as JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    print(f"Saved {len(records)} predictions to {output_path}")
    return output_path


def summarise_prediction_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute a lightweight pipeline summary for console logging."""
    answered = [r for r in records if not r.get("abstain", False) and not r.get("error")]
    abstained = [r for r in records if r.get("abstain", False) and not r.get("error")]
    errored = [r for r in records if r.get("error")]

    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    total_tokens = sum((r.get("total_tokens") or 0) for r in records)

    return {
        "n_total": len(records),
        "n_answered": len(answered),
        "n_abstained": len(abstained),
        "n_errors": len(errored),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "total_tokens": total_tokens,
    }


def run_full_pipeline(
    samples: pd.DataFrame,
    models: List[str] | None = None,
    output_dir: str | Path = "outputs",
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Run all configured models across the sampled dataset and save JSONL outputs.

    Returns a mapping of model name to prediction file path.
    """
    model_names = models or list(DEFAULT_MODELS)
    output_dir = Path(output_dir)
    client = OpenAI()
    output_paths: Dict[str, Path] = {}

    for model in model_names:
        print(f"\n{'=' * 60}")
        print(f"Running model: {model}")
        print(f"{'=' * 60}")

        records = run_model_pipeline(
            client=client,
            model=model,
            samples=samples,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
            verbose=verbose,
        )

        output_path = output_dir / f"predictions_{_safe_model_name(model)}.jsonl"
        save_predictions(records, output_path)
        output_paths[model] = output_path

        summary = summarise_prediction_records(records)
        print(f"\nSummary for {model}:")
        print(f"  Answered:     {summary['n_answered']}/{summary['n_total']}")
        print(f"  Abstained:    {summary['n_abstained']}/{summary['n_total']}")
        print(f"  Errors:       {summary['n_errors']}/{summary['n_total']}")
        latency_text = (
            f"{summary['mean_latency_ms']:.0f}ms"
            if summary["mean_latency_ms"] is not None
            else "n/a"
        )
        print(f"  Avg latency:  {latency_text}")
        print(f"  Total tokens: {summary['total_tokens']:,}")

    return output_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the Financial QA pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample progress.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    args = parse_args()
    config = load_config(args.config)

    dataset_cfg = config["dataset"]
    pipeline_cfg = config["pipeline"]
    output_cfg = config["output"]
    model_names = extract_model_names(config)

    from src.data_loader import load_sample

    csv_path = dataset_cfg["csv_path"]
    sample_size = int(dataset_cfg["sample_size"])
    seed = int(dataset_cfg["random_seed"])

    print("Loading dataset and sampling rows...")
    samples = load_sample(csv_path=csv_path, n=sample_size, seed=seed)
    print(
        f"Loaded {len(samples)} samples across "
        f"{samples['ticker'].nunique()} tickers"
    )

    output_paths = run_full_pipeline(
        samples=samples,
        models=model_names,
        output_dir=output_cfg["predictions_dir"],
        temperature=float(pipeline_cfg["temperature"]),
        max_retries=int(pipeline_cfg["max_retries"]),
        retry_delay=float(pipeline_cfg["retry_delay_seconds"]),
        verbose=args.verbose,
    )

    print("\n=== DONE ===")
    for model, path in output_paths.items():
        print(f"  {model}: {path}")


if __name__ == "__main__":
    main()
