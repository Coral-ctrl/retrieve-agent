"""Settings loaded from environment variables (and a local .env file).

Keeping configuration in one place means the rest of the code never reads
os.environ directly, which makes it easy to test and easy to explain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Default model for summarising and tagging. Sonnet is a good balance of
# quality and cost for this job; switch to Haiku in .env to save money.
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TURNS = 20


# Which way to reach Claude. "subscription" runs the agent through the
# Claude Agent SDK, signed in with your Claude Pro/Max plan, so its usage is
# covered by the plan's monthly credit. "api" calls the Anthropic API with a
# pay-per-token key. With "subscription", the API key (if set) is only the
# fallback for when the subscription path can't run.
BACKEND_SUBSCRIPTION = "subscription"
BACKEND_API = "api"
DEFAULT_BACKEND = BACKEND_SUBSCRIPTION


@dataclass(frozen=True)
class Settings:
    backend: str  # BACKEND_SUBSCRIPTION or BACKEND_API
    anthropic_api_key: str | None
    claude_oauth_token: str | None  # from `claude setup-token`; optional if `claude` is logged in
    model: str
    max_turns: int
    knowledge_dir: Path  # where distilled notes are written
    logs_dir: Path  # where agent run logs are written
    database_url: str | None  # Postgres/Neon connection string; DB sync is skipped if unset
    voyage_api_key: str | None  # needed to embed topics for de-duplication
    embedding_model: str
    dedup_threshold: float  # cosine distance below which two topics are treated as one


def load_settings(project_root: Path | None = None) -> Settings:
    """Read settings from the environment, loading .env first if present."""
    root = project_root or Path.cwd()
    load_dotenv(root / ".env")

    backend = os.getenv("RETRIEVE_BACKEND", DEFAULT_BACKEND).strip().lower()
    if backend not in {BACKEND_SUBSCRIPTION, BACKEND_API}:
        raise ValueError(
            f"RETRIEVE_BACKEND must be '{BACKEND_SUBSCRIPTION}' or '{BACKEND_API}', not '{backend}'."
        )

    return Settings(
        backend=backend,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        claude_oauth_token=os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or None,
        model=os.getenv("RETRIEVE_MODEL", DEFAULT_MODEL),
        max_turns=int(os.getenv("RETRIEVE_MAX_TURNS", DEFAULT_MAX_TURNS)),
        knowledge_dir=root / os.getenv("RETRIEVE_KNOWLEDGE_DIR", "knowledge"),
        logs_dir=root / os.getenv("RETRIEVE_LOGS_DIR", "logs"),
        database_url=os.getenv("DATABASE_URL") or None,
        voyage_api_key=os.getenv("VOYAGE_API_KEY") or None,
        embedding_model=os.getenv("RETRIEVE_EMBEDDING_MODEL", "voyage-4-lite"),
        dedup_threshold=float(os.getenv("RETRIEVE_DEDUP_THRESHOLD", "0.10")),
    )
