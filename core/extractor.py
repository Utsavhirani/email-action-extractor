"""Single public entry point: picks the LLM or the stub extractor based on USE_LLM."""

from core import config
from core.models import Email, ExtractedAction
from core.stub_extractor import extract_actions_stub


def active_mode() -> str:
    """'stub', or the main LLM model ID when USE_LLM=1. Used as the eval report label."""
    return config.llm_model() if config.use_llm() else "stub"


def extract_actions(email: Email) -> list[ExtractedAction]:
    """Extract action points from one email."""
    if config.use_llm():
        # Imported lazily so stub mode never loads the HTTP client.
        from core import llm_extractor

        return llm_extractor.extract_actions_llm(email)
    return extract_actions_stub(email)
