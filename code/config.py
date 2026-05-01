from __future__ import annotations

import os
import random
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent.parent
DATA_DIR: Path = BASE_DIR / "data"
SUPPORT_TICKETS_DIR: Path = BASE_DIR / "support_tickets"
CHROMA_DIR: Path = Path(__file__).parent / ".chroma_db"

# ── Load .env via dotenv_values (reads file directly, strips quotes reliably) ──
_env: dict[str, str] = {}
_env_path = BASE_DIR / ".env"
try:
    from dotenv import dotenv_values
    if _env_path.exists():
        _env = {k: v for k, v in dotenv_values(_env_path).items() if v is not None}
except ImportError:
    pass


def _get(key: str, default: str = "") -> str:
    """Prefer process env vars, then .env file values, then default."""
    return os.environ.get(key) or _env.get(key) or default


# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = _get("LLM_PROVIDER", "openai")
ANTHROPIC_API_KEY: str = _get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = _get("OPENAI_API_KEY", "")
ANTHROPIC_MODEL: str = _get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
OPENAI_MODEL: str = _get("OPENAI_MODEL", "gpt-4o")
LLM_TEMPERATURE: float = float(_get("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(_get("LLM_MAX_TOKENS", "1500"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = _get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = int(_get("TOP_K", "5"))
CHUNK_SIZE: int = int(_get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(_get("CHUNK_OVERLAP", "100"))

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED: int = int(_get("RANDOM_SEED", "42"))
random.seed(RANDOM_SEED)

# ── Collection name ───────────────────────────────────────────────────────────
CHROMA_COLLECTION: str = "support_corpus_v1"
