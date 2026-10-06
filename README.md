# Chidi

An independent Socratic learning-companion prototype built for
LLMOps learning and interview preparation.

The project is inspired by public descriptions of ALX's Chidi.
It does not reproduce ALX's internal implementation.

## Features

- Pydantic-validated learner inputs and tutor replies.
- LangGraph routing and in-process conversation memory.
- Mock tutoring for free, repeatable checks.
- Live tutoring through OpenRouter.
- LangSmith tracing through the dedicated trace demo.
- Local Qdrant retrieval with FastEmbed embeddings.
- JSONL datasets and per-case evaluation reports.
- Rule-based answer-disclosure checks.
- A structured pedagogical LLM judge.
- RAGAS faithfulness measurements with explicit evidence scopes.
- pandas summaries and GitHub Actions CI.

## Setup

Use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,live,analysis,retrieval,evaluation]" -c requirements.lock.txt
python -m pip check
```

Create a local `.env` using `.env.example` as a guide.
Configure these values for live features:

- `ANTHROPIC_API_KEY`: OpenRouter API key for the configured endpoint.
- `ANTHROPIC_BASE_URL`: [https://openrouter.ai/api](https://openrouter.ai/api)
- `ANTHROPIC_MODEL`: exact model ID accepted by the tutor endpoint.
- `CHIDI_JUDGE_MODEL`: model ID accepted by the pedagogical judge endpoint.
- `RAGAS_MODEL`: exact model ID from the OpenRouter `/api/v1` catalogue.
- `LANGSMITH_API_KEY`: LangSmith API key.
- `LANGSMITH_PROJECT`: `chidi`.
- `LANGSMITH_ENDPOINT`: endpoint matching the LangSmith account region.

Model availability depends on the endpoint.
Secrets belong in `.env`, which is excluded from Git.

## Free checks

```bash
python -m pytest -q
python scripts/run_eval.py
python scripts/run_behavior_eval.py
python scripts/demo_regression.py
```

## Retrieval

```bash
python scripts/ingest.py
python scripts/run_retrieval_eval.py
python scripts/search_concepts.py "The learner forgot the afternoon round trip."
```

The first ingestion downloads the embedding model.
Qdrant data is stored locally under `data/processed/`.
Run one retrieval process at a time.

## Live tutoring and tracing

```bash
python scripts/chat.py --live
python scripts/trace_demo.py
```

Live model requests incur charges.
Conversation memory lasts for the current Python process.
The trace demo explicitly enables LangSmith tracing.

## Evaluation

The pedagogical judge evaluates accuracy, misconception handling,
question load and premature disclosure separately.

RAGAS checks evidence support. The demo distinguishes retrieved
concepts alone from concepts plus learner-provided information.

```bash
python scripts/run_ragas_demo.py
```

The judge and reporting scripts consume locally saved reports.
Use their `--help` output for file-selection arguments.

## Data provenance

MathDial candidates are prepared from a separately downloaded dataset.
Source identifiers and the raw-file hash are retained.
Original candidates remain separate from project review decisions.

Some inspected cases have ambiguous wording or questionable references.
Those cases are held out of automatic correctness scoring.

Concept notes are project-authored.
Synthetic judge calibration responses are explicitly labelled.
AI-proposed review labels require human verification before being
described as human ground truth.

Raw data, processed data and reports are excluded from Git.

## Measured development results

- 25 local tests passed at the documented checkpoint.
- Four retrieval development queries returned the expected concept first.
- A controlled mock disclosure fault failed two prohibited-disclosure cases.
- Under the same judge rubric, single-question passes increased from
  0/4 to 2/4 after a prompt change.
- The revised prompt retained four accuracy and misconception passes,
  with no premature disclosures flagged on those four examples.
- One RAGAS demonstration scored 0.00 against concept-only evidence
  and 0.75 against combined concept and learner evidence.

These are small development checks.
They do not establish general tutor effectiveness or learner mastery.

## Remaining work

- Broader independently reviewed evaluation data and untouched test cases.
- Unknown-failure discovery through fresh interaction review.
- Langfuse integration.
- Paid evaluation workflow with explicit budget controls.
- Optional FastAPI service and deployment.
