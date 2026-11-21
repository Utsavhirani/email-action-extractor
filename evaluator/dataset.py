"""Golden dataset: emails paired with the actions a correct extractor should return."""

from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from core.models import Email, ExtractedAction

DEFAULT_DATASET_PATH = Path(__file__).parent / "data" / "golden.json"


class GoldenCase(BaseModel):
    """One test email and its expected actions. `notes` explains tricky labels."""

    id: str
    email: Email
    expected: list[ExtractedAction]
    notes: str = ""


_CASES_ADAPTER = TypeAdapter(list[GoldenCase])


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> list[GoldenCase]:
    """Load and validate a JSON list of cases. Raises on invalid data or duplicate ids."""
    cases = _CASES_ADAPTER.validate_json(Path(path).read_text(encoding="utf-8"))
    ids = [case.id for case in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate case ids: {duplicates}")
    return cases
