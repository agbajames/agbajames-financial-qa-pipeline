# Financial QA Pipeline

A lightweight, context-only LLM question-answering pipeline over Financial 10-K filings, comparing `gpt-4.1` and `gpt-4.1-mini` on quality, calibration, latency, and cost.

The pipeline runs both models on a stratified sample of 50 questions, saves structured predictions, and produces an evaluation that informs a production model recommendation.

## Quick Start

### Prerequisites

- Python 3.10+ (tested on 3.11)
- An OpenAI API key with access to `gpt-4.1` and `gpt-4.1-mini`

### Setup

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
echo "OPENAI_API_KEY=sk-..." > .env
```

### Run

**Option A – notebook (recommended for review)**

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook is the main deliverable. It loads the dataset, runs both models, evaluates predictions, and renders the comparison charts and recommendation. Predictions are cached to `outputs/predictions_*.jsonl` on the first run, so subsequent runs do not re-call the API.

A pre-rendered, viewable copy is also available at `analysis.html` (no Python required).

**Option B – CLI**

```bash
python -m src.pipeline --config config.yaml --verbose
```

Runs both models and writes JSONL predictions to `outputs/`.

### Estimated Cost

Running both models live on 50 samples costs **~$0.14 total** (`gpt-4.1` ≈ $0.12, `gpt-4.1-mini` ≈ $0.02), based on OpenAI's per-million-token list prices: $2.00 / $8.00 (input/output) for `gpt-4.1` and $0.40 / $1.60 for `gpt-4.1-mini`.

## Results

Evaluated on 50 stratified samples (seed = 42) across 43 tickers, all 2023 10-K filings.

| Metric | gpt-4.1 | gpt-4.1-mini |
|---|---|---|
| Coverage | 96% | 96% |
| Exact Match | 25% | **29%** |
| Token F1 | 0.730 | **0.767** |
| ROUGE-L | 0.712 | **0.752** |
| Numerical accuracy | 75% | **83%** |
| Should-have-abstained | 6% | **2%** |
| Avg latency (ms) | **1,771** | 2,705 |
| Cost / 1K queries | $2.32 | **$0.47** |

**Recommendation: `gpt-4.1-mini`.** It matches or exceeds `gpt-4.1` on every quality metric on this sample (Token F1, ROUGE-L, Exact Match, numerical accuracy, coverage) at roughly one-fifth the cost per query. `gpt-4.1`'s only advantage is latency (~930 ms / 53% faster), which would matter for latency-sensitive use cases but is unlikely to be material for back-office analyst workflows.

The full reasoning, including caveats for production rollout, is in section 8 of the notebook and in `slides/approach_summary.pptx`.

## Project Structure

```
financial-qa-pipeline/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.yaml                        # Models, paths, pipeline settings
├── Financial-QA-10k.csv               # Dataset (Kaggle)
├── analysis.html                      # Rendered notebook (no Python required)
├── notebooks/
│   └── analysis.ipynb                 # Main analysis notebook
├── src/
│   ├── data_loader.py                 # Stratified sampling
│   ├── prompt_and_schema.py           # System prompt + JSON schema + post-processing
│   ├── pipeline.py                    # OpenAI calls + JSONL output
│   └── evaluation.py                  # Lexical, numerical, and abstention metrics
├── outputs/                           # Generated artefacts
│   ├── samples.csv                    # Frozen 50-row sample
│   ├── predictions_*.jsonl            # Per-model prediction records
│   ├── eval_summary_*.json            # Aggregate metrics per model
│   ├── eval_details_*.csv             # Per-sample evaluation
│   ├── model_comparison.csv           # Side-by-side comparison
│   ├── recommendation.json            # Recommended model + rationale
│   └── *.png                          # Charts used in the slides
└── slides/
    └── approach_summary.pptx          # 4-slide summary
```

## Design Decisions

### Sampling

The dataset has 6,997 rows across 69 tickers, all from 2023 10-K filings. The brief allows taking the first 50 rows, but doing so would yield 50 NVIDIA-only questions (all `long_text` and `short_text` answers, with only 2 numerics) – not representative. Instead, the sample is **stratified** with `seed=42` and a minimum-per-type floor of 8:

- **8 numeric** answers (currency, percentages, figures)
- **8 short text** answers (< 60 characters)
- **34 long text** answers (explanatory)
- **43 unique tickers**

This trades a small departure from the literal "first 50" rule for a much more useful evaluation set – numeric answers exercise the numerical accuracy metric, short answers exercise exact match, and long answers exercise overlap-based metrics.

### Prompt and structured outputs

The system prompt enforces three things:

1. **Context-only answering.** The model must answer from the provided 10-K snippet, not parametric memory.
2. **Explicit abstention.** If the context does not support a reliable answer, the model must abstain (return `abstain: true` with an empty `answer` and a reason code).
3. **Evidence quotes.** Every answered response must include exact quotes from the context, which a post-processing step verifies by character-span matching.

Outputs use OpenAI's JSON schema mode (Structured Outputs) rather than free-form JSON parsing, so every response is guaranteed to match the schema or fail the API call.

### Evaluation

Three metric families, each addressing a different production concern:

- **Lexical overlap** – Exact Match, Token F1 (SQuAD-style), ROUGE-L (LCS-based, computed in-house). Captures how closely the answer matches the reference text.
- **Numerical correctness** – Number extraction and comparison against reference values for figure-based answers. Captures whether the figures are right regardless of phrasing.
- **Operational** – Coverage, should-have-abstained rate, average latency, projected cost per 1K queries.

A "should-have-abstained" diagnostic flags answered cases with very low overlap; this is treated as a calibration signal, not a primary decision metric, because concise correct numeric answers can also score low on lexical overlap.

### Model selection

Two models from the same family, to isolate the quality-versus-cost trade-off cleanly:

| Model | Input / 1M tokens | Output / 1M tokens |
|---|---|---|
| `gpt-4.1` | $2.00 | $8.00 |
| `gpt-4.1-mini` | $0.40 | $1.60 |

Same architecture, same API surface, ~5× cost difference – the question becomes "is the cheaper model good enough?" rather than "which family is best?"

## Limitations

- **Sample size.** 50 samples gives directional rather than statistically robust comparisons. The 8-percentage-point numerical accuracy gap, the 4-point Exact Match gap, and the 1-sample should-have-abstained gap should all be treated as directional. A larger evaluation set (~500 questions, stratified by ticker and answer type) would be needed before deployment.
- **Lexical metric bias against concise correct answers.** Token F1, ROUGE-L, and Exact Match penalise concise correct answers when the reference is a full sentence – a recurring pattern in the failure-case analysis. Numerical accuracy partially compensates for figure-based questions; for free-text answers a semantic-similarity metric or LLM-judge layer would help.
- **Filing scope.** All 50 rows are from 2023 10-Ks. Results may not generalise to other years or filing types.
- **Determinism.** OpenAI API outputs at temperature 0 are mostly but not fully deterministic; metrics may vary slightly across runs due to non-determinism on OpenAI's side.

## References

- **Dataset:** [Financial QA 10-K on Kaggle](https://www.kaggle.com/datasets/yousefsaeedian/financial-q-and-a-10k)
