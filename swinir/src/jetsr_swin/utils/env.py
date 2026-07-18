from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_from_repo(start: Path | None = None) -> Path | None:
    """Find and load the nearest .env walking up from `start` (default: this file).

    Returns the path that was loaded, or None if none found. Silent if
    python-dotenv is missing.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return candidate
    return None


def require_wandb_env() -> None:
    """Raise a helpful error if W&B env is incomplete."""
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY not set. Put it in swinir/.env or export it, "
            "or pass --no-wandb to skip logging."
        )
