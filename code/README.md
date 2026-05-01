# Multi-Domain Support Triage Agent

A production-quality RAG-powered support triage agent covering **HackerRank**, **Claude**, and **Visa** — built for the HackerRank Orchestrate hackathon.

---

## Architecture

```
Ticket Input
     │
     ▼
┌─────────────┐    ┌──────────────────────────────────────────────┐
│ safety.py   │    │ Corpus (data/)                               │
│ ─────────── │    │   hackerrank/ (436 articles)                 │
│ • injection │    │   claude/     (319 articles)                 │
│ • escalation│    │   visa/       (14 articles)                  │
│   keywords  │    └──────────┬───────────────────────────────────┘
└──────┬──────┘               │
       │              ┌───────▼──────────┐
       │              │  ingestion.py    │
       │              │  ─────────────── │
       │              │  • parse MD      │
       │              │  • chunk (800ch) │
       │              │  • embed (MiniLM)│
       │              │  • ChromaDB      │
       │              └───────┬──────────┘
       │                      │
       ▼              ┌───────▼──────────┐
┌─────────────┐       │  retriever.py    │
│ agent.py    │◄──────│  ─────────────── │
│ ─────────── │       │  • company filter│
│ orchestrate │       │  • top-k=5 docs  │
│ → LLM call  │       │  • cosine sim    │
│ → parse JSON│       └──────────────────┘
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ output.csv  │
│ ─────────── │
│ status      │
│ product_area│
│ response    │
│ justification│
│ request_type│
└─────────────┘
```

### Key design decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| LLM | Claude claude-sonnet-4-5 / GPT-4o | Structured tool-use output, strong reasoning |
| Embeddings | `all-MiniLM-L6-v2` | Fast, offline, good semantic quality |
| Vector DB | ChromaDB (persistent) | Local, no server, metadata filtering |
| Output schema | Pydantic + Anthropic tool-use | Type-safe, deterministic JSON |
| Safety | Deterministic regex + LLM | Defense in depth |

---

## Setup

### 1. Install dependencies

```bash
pip install -r ../requirements.txt
```

> **Python 3.11+ recommended.** The sentence-transformers model (~90 MB) is downloaded on first run.

### 2. Configure environment

```bash
cp ../.env.example ../.env
# Edit .env and set ANTHROPIC_API_KEY or OPENAI_API_KEY
```

`.env` variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` | — | Required if using Anthropic |
| `OPENAI_API_KEY` | — | Required if using OpenAI |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Model name |
| `OPENAI_MODEL` | `gpt-4o` | Model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `TOP_K` | `5` | Retrieval top-k |
| `LLM_TEMPERATURE` | `0.0` | Deterministic output |

---

## Usage

All commands are run from the `code/` directory.

### Batch mode (main use case)

```bash
python main.py
```

This processes `support_tickets/support_tickets.csv` and writes to `support_tickets/output.csv`.

With custom paths:

```bash
python main.py --input ../support_tickets/support_tickets.csv \
               --output ../support_tickets/output.csv
```

With debug output (shows retrieved docs + justification per ticket):

```bash
python main.py --debug
```

### Single ticket mode

```bash
python main.py --ticket "I lost my Visa card in Portugal" --company Visa
```

### Ingest-only (pre-build the index)

```bash
python main.py --ingest
```

Force re-index:

```bash
python main.py --ingest --force-reindex
```

### Evaluation mode (compare against sample CSV)

```bash
python main.py --evaluate
```

---

## Module responsibilities

| File | Role |
|------|------|
| `main.py` | CLI entry point, Rich terminal UI, batch/single/eval modes |
| `agent.py` | Orchestrator: safety → retrieval → LLM → output |
| `ingestion.py` | Corpus loading, markdown parsing, chunking, ChromaDB indexing |
| `retriever.py` | Semantic search with company-aware filtering |
| `safety.py` | Deterministic injection detection + escalation keyword matching |
| `prompts.py` | All system/user prompts and tool schemas |
| `schemas.py` | Pydantic models for all data types |
| `config.py` | Environment-driven configuration |
| `evaluator.py` | Accuracy metrics, confusion matrix, row-level diff |

---

## Safety & escalation logic

**Layer 1 — Deterministic (pre-LLM):**
- Regex patterns for prompt injection (`ignore previous instructions`, `reveal system prompt`, `DAN`, etc.)
- Keyword matching for: fraud, identity theft, account compromise, legal threats, billing disputes, security vulnerabilities, score manipulation

**Layer 2 — LLM reasoning:**
- System prompt instructs the LLM to escalate when documentation is insufficient
- Safety flags from Layer 1 are passed as hints to the LLM
- Tool-use schema enforces valid enum values

**Layer 3 — Fallback:**
- Any LLM exception → safe escalation response

---

## Reproducibility

- `LLM_TEMPERATURE=0.0` (deterministic sampling)
- `RANDOM_SEED=42`
- ChromaDB index is persisted in `code/.chroma_db/` — re-runs skip re-indexing

---

## Output schema

| Column | Allowed values |
|--------|---------------|
| `status` | `replied`, `escalated` |
| `product_area` | Free text (e.g. `screen`, `privacy`, `general_support`) |
| `response` | User-facing answer grounded in corpus |
| `justification` | Internal routing rationale |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` |
