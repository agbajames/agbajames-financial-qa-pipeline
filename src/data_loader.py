# file: src/data_loader.py

"""
Data loader for the Financial QA 10-K dataset.

This module loads the source CSV, classifies answers into coarse answer
types, and draws a reproducible 50-row stratified sample.

Why stratified sampling:
- The brief allows using the first 50 rows, but a simple first-50 slice can
  over-represent one company or one answer style.
- A stratified sample gives a more representative evaluation set across
  answer types and tickers while remaining lightweight and reproducible.

Answer types:
- numeric
- short_text
- long_text
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH: Final[Path] = ROOT / "Financial-QA-10k.csv"
ANSWER_TYPES: Final[tuple[str, str, str]] = ("numeric", "short_text", "long_text")
REQUIRED_COLUMNS: Final[set[str]] = {"question", "answer", "context", "ticker"}
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("filing",)

_BASE_NUMBER_PATTERN = r"(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"
_MAGNITUDE_PATTERN = r"(?:billion|million|thousand|trillion|[BMKT])\b"
_UNIT_PATTERN = r"(?:%|percent|basis points?|bps|shares?|x|times?)"
_QUALIFIER_PATTERN = (
    r"(?:about|approximately|approx\.?|around|over|under|nearly|roughly|"
    r"more than|less than)"
)
_COMPACT_NUMERIC_PATTERN = re.compile(
    rf"""
    ^
    \s*
    (?:{_QUALIFIER_PATTERN}\s+)?
    (?:
        \(
            \s*[$€£]?\s*{_BASE_NUMBER_PATTERN}
            (?:\s*{_MAGNITUDE_PATTERN})?
            (?:\s*{_UNIT_PATTERN})?
            \s*
        \)
        |
        (?:
            -?\s*[$€£]?\s*{_BASE_NUMBER_PATTERN}
            |
            [$€£]\s*-?\s*{_BASE_NUMBER_PATTERN}
        )
        (?:\s*{_MAGNITUDE_PATTERN})?
        (?:\s*{_UNIT_PATTERN})?
    )
    (?:\s+(?:in|for|per|at|of)\s+[A-Za-z][\w/-]*)*
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def classify_answer_type(answer: str) -> str:
    """Classify an answer string into numeric, short_text, or long_text."""
    text = re.sub(r"\s+", " ", str(answer)).strip()
    if not text:
        return "short_text"

    word_count = len(re.findall(r"\w+", text))

    if word_count <= 8 and _COMPACT_NUMERIC_PATTERN.fullmatch(text):
        return "numeric"
    if len(text) < 60:
        return "short_text"
    return "long_text"


def _allocate_targets(
    df: pd.DataFrame,
    n: int,
    target_ratios: dict[str, float] | None = None,
    min_per_type: int = 8,
) -> dict[str, int]:
    """Allocate per-answer-type targets that sum to n and respect capacity."""
    available = (
        df["answer_type"]
        .value_counts()
        .reindex(ANSWER_TYPES, fill_value=0)
        .astype(int)
    )

    if target_ratios is None:
        ratios = (
            available.astype(float) / float(available.sum())
            if int(available.sum()) > 0
            else pd.Series(0.0, index=ANSWER_TYPES)
        )
    else:
        ratios = pd.Series(target_ratios, dtype=float).reindex(
            ANSWER_TYPES,
            fill_value=0.0,
        )
        total = float(ratios.sum())
        if total <= 0:
            raise ValueError("target_ratios must sum to a positive value.")
        ratios = ratios / total

    present_types = [
        answer_type
        for answer_type in ANSWER_TYPES
        if available[answer_type] > 0 and ratios[answer_type] > 0
    ]

    if not present_types:
        present_types = [
            answer_type for answer_type in ANSWER_TYPES if available[answer_type] > 0
        ]

    if not present_types:
        return {answer_type: 0 for answer_type in ANSWER_TYPES}

    if int(available[present_types].sum()) < n:
        raise ValueError(
            f"Requested n={n}, but eligible answer types only provide "
            f"{int(available[present_types].sum())} rows."
        )

    raw_targets = {
        answer_type: float(ratios[answer_type]) * n
        for answer_type in present_types
    }
    targets = {
        answer_type: min(
            int(available[answer_type]),
            int(np.floor(raw_targets[answer_type])),
        )
        for answer_type in present_types
    }

    minimum_targets = {
        answer_type: min(min_per_type, int(available[answer_type]))
        for answer_type in present_types
    }
    enforce_minimum = sum(minimum_targets.values()) <= n

    if enforce_minimum:
        for answer_type in present_types:
            targets[answer_type] = max(
                targets[answer_type],
                minimum_targets[answer_type],
            )

    while sum(targets.values()) > n:
        candidates = [
            answer_type
            for answer_type in present_types
            if targets[answer_type] > (
                minimum_targets[answer_type] if enforce_minimum else 0
            )
        ]
        if not candidates:
            candidates = [
                answer_type
                for answer_type in present_types
                if targets[answer_type] > 0
            ]
        chosen = max(
            candidates,
            key=lambda answer_type: targets[answer_type] - raw_targets[answer_type],
        )
        targets[chosen] -= 1

    while sum(targets.values()) < n:
        candidates = [
            answer_type
            for answer_type in present_types
            if targets[answer_type] < int(available[answer_type])
        ]
        if not candidates:
            break
        chosen = max(
            candidates,
            key=lambda answer_type: (
                raw_targets[answer_type] - targets[answer_type],
                int(available[answer_type]) - targets[answer_type],
            ),
        )
        targets[chosen] += 1

    return {
        answer_type: targets.get(answer_type, 0)
        for answer_type in ANSWER_TYPES
    }


def stratified_sample(
    df: pd.DataFrame,
    n: int = 50,
    seed: int = 42,
    target_ratios: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Draw a stratified sample of n rows, balanced across answer types and
    diversified across tickers.
    """
    if n <= 0:
        raise ValueError("n must be greater than 0.")
    if len(df) < n:
        raise ValueError(f"Requested n={n}, but dataset only has {len(df)} rows.")

    rng_state = np.random.RandomState(seed)
    targets = _allocate_targets(df, n=n, target_ratios=target_ratios, min_per_type=8)

    sampled_parts: list[pd.DataFrame] = []
    used_indices: set[int] = set()

    for answer_type, count in targets.items():
        if count <= 0:
            continue

        pool = df[
            (df["answer_type"] == answer_type)
            & (~df.index.isin(used_indices))
        ]
        if pool.empty:
            continue

        tickers = list(pool["ticker"].dropna().astype(str).unique())
        rng_state.shuffle(tickers)

        type_samples: list[pd.DataFrame] = []

        for ticker in tickers:
            if len(type_samples) >= count:
                break
            ticker_pool = pool[pool["ticker"] == ticker]
            if ticker_pool.empty:
                continue
            row = ticker_pool.sample(n=1, random_state=rng_state)
            type_samples.append(row)

        picked_idx = (
            pd.Index([part.index[0] for part in type_samples])
            if type_samples
            else pd.Index([])
        )
        remaining_pool = pool[~pool.index.isin(picked_idx)]
        still_needed = count - len(type_samples)

        if still_needed > 0 and not remaining_pool.empty:
            extra = remaining_pool.sample(
                n=min(still_needed, len(remaining_pool)),
                random_state=rng_state,
            )
            type_samples.append(extra)

        if type_samples:
            part = pd.concat(type_samples, axis=0)
            used_indices.update(part.index.tolist())
            sampled_parts.append(part)

    result = (
        pd.concat(sampled_parts, axis=0)
        if sampled_parts
        else df.iloc[0:0].copy()
    )

    shortfall = n - len(result)
    if shortfall > 0:
        remaining_pool = df[~df.index.isin(used_indices)]
        if len(remaining_pool) < shortfall:
            raise ValueError(
                f"Sampling shortfall of {shortfall}, but only "
                f"{len(remaining_pool)} rows remain."
            )
        backfill = remaining_pool.sample(n=shortfall, random_state=rng_state)
        result = pd.concat([result, backfill], axis=0)

    result = result.sample(frac=1, random_state=rng_state).reset_index(drop=True)
    result.insert(0, "sample_id", range(len(result)))
    return result


def load_dataset(csv_path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """
    Load the full dataset, validate columns, drop null text fields, and add
    answer_type.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df.dropna(subset=["question", "answer", "context"]).reset_index(drop=True)
    df["answer_type"] = df["answer"].apply(classify_answer_type)
    return df


def load_sample(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    n: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Load the dataset and return a reproducible stratified sample."""
    df = load_dataset(csv_path)
    return stratified_sample(df, n=n, seed=seed)


if __name__ == "__main__":
    import sys

    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    sample = load_sample(csv_path)

    print(f"Sample shape: {sample.shape}")
    print(f"\nAnswer type distribution:\n{sample['answer_type'].value_counts()}")
    print(f"\nTicker diversity: {sample['ticker'].nunique()} unique tickers")
    print(f"Tickers: {sorted(sample['ticker'].dropna().astype(str).unique())}")
    print(f"\nContext length stats (chars):\n{sample['context'].str.len().describe()}")
    print(f"\nAnswer length stats (chars):\n{sample['answer'].str.len().describe()}")

    print("\n=== SAMPLE PREVIEW ===")
    for _, row in sample.head(5).iterrows():
        print(f"\n[{row['sample_id']}] ({row['ticker']}, {row['answer_type']})")
        print(f"  Q: {row['question'][:100]}")
        print(f"  A: {row['answer'][:100]}")
