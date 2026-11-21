"""All environment-variable settings, read in one place.

Settings come from the project's `.env` file (copy `.env.example` to start). Variables
already set in the shell win over `.env`, so `$env:USE_LLM="0"` still works as a one-off.
Values are read on every call (not cached at import) so tests can monkeypatch them.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_FILE, override=False)  # missing file is fine: defaults apply

# Free OpenRouter models that support structured (JSON-schema) output. Checked against
# GET https://openrouter.ai/api/v1/models on 2026-11-21 (pricing 0, "structured_outputs"
# in supported_parameters). Free models come and go: re-check that list if one disappears.
DEFAULT_LLM_MODEL = "nvidia/nemotron-3.5-super-vl:free"
DEFAULT_LLM_FALLBACK_MODELS = "inclusionai/ling-3.2-flash:free,baidu/ernie-5.5-flash-preview:free"
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0


def use_llm() -> bool:
    """True only when USE_LLM is exactly '1'. Unset or anything else means off."""
    return os.environ.get("USE_LLM", "").strip() == "1"


def openrouter_api_key() -> str:
    """API key from OPENROUTER_API_KEY. Raises a clear error if it's missing."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Get a free key at https://openrouter.ai/keys "
            "and put it in .env"
        )
    return key


def llm_model() -> str:
    """Main model ID from LLM_MODEL, or the default free model."""
    return os.environ.get("LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL


def llm_fallback_models() -> list[str]:
    """Comma-separated LLM_FALLBACK_MODELS, or the defaults. 'none' disables fallback."""
    raw = os.environ.get("LLM_FALLBACK_MODELS", "").strip() or DEFAULT_LLM_FALLBACK_MODELS
    if raw.lower() == "none":
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


def llm_models() -> list[str]:
    """The main model followed by the fallbacks, without duplicates."""
    return list(dict.fromkeys([llm_model(), *llm_fallback_models()]))


def llm_timeout_seconds() -> float:
    """Max seconds to wait for one response (LLM_TIMEOUT_SECONDS, default 60)."""
    raw = os.environ.get("LLM_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"LLM_TIMEOUT_SECONDS must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"LLM_TIMEOUT_SECONDS must be positive, got {raw!r}")
    return value


def llm_call_delay_seconds() -> float:
    """Pause after each LLM call (LLM_CALL_DELAY_SECONDS, default 0, never negative)."""
    raw = os.environ.get("LLM_CALL_DELAY_SECONDS", "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError as exc:
        raise ValueError(f"LLM_CALL_DELAY_SECONDS must be a number, got {raw!r}") from exc
