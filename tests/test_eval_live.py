"""Live eval over golden.json. Skipped by default; run with:  pytest -m eval -s

USE_LLM off -> stub baseline (no API).  USE_LLM=1 -> real OpenRouter calls (16 emails = 16+ of the ~50 free requests/day).
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.extractor import active_mode, extract_actions
from core.prompts import PROMPT_VERSION
from evaluator.dataset import load_dataset
from evaluator.runner import evaluate

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

# Placeholder thresholds: set real ones after looking at the first runs.
MIN_RECALL = 0.0
MIN_PRECISION = 0.0


def run_filename(mode: str) -> str:
    """e.g. 20261121T140444Z_nvidia_nemotron-3.5-super-vl_free_v1.json (no characters Windows rejects)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_mode = re.sub(r"[^A-Za-z0-9._-]", "_", mode)
    return f"{stamp}_{safe_mode}_{PROMPT_VERSION}.json"


@pytest.mark.eval
def test_live_eval() -> None:
    mode = active_mode()
    report = evaluate(load_dataset(), extract_actions, mode=mode, prompt_version=PROMPT_VERSION)
    path = report.save(RUNS_DIR / run_filename(mode))

    print("\n" + report.summary())
    print(f"\nSaved report to {path}")

    assert report.recall >= MIN_RECALL
    assert report.precision >= MIN_PRECISION
