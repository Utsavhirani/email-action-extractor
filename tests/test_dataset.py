"""Tests for evaluator.dataset and the golden.json file itself."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from core.models import ActionType, normalize_address
from evaluator.dataset import GoldenCase, load_dataset


@pytest.fixture(scope="module")
def cases() -> list[GoldenCase]:
    return load_dataset()


def test_golden_loads_and_validates(cases: list[GoldenCase]) -> None:
    assert 14 <= len(cases) <= 20
    assert all(isinstance(c, GoldenCase) for c in cases)


def test_expected_owners_are_normalized_participants(cases: list[GoldenCase]) -> None:
    for case in cases:
        for action in case.expected:
            if action.owner is not None:
                assert action.owner == normalize_address(action.owner), case.id
                assert action.owner in case.email.participants(), case.id


def test_due_dates_not_before_email(cases: list[GoldenCase]) -> None:
    for case in cases:
        for action in case.expected:
            if action.due_date is not None:
                assert action.due_date >= case.email.date.date(), case.id


def test_dataset_covers_required_scenarios(cases: list[GoldenCase]) -> None:
    all_types = {a.action_type for c in cases for a in c.expected}
    assert any(len(c.expected) == 1 for c in cases), "single action"
    assert any(len(c.expected) >= 2 for c in cases), "several actions"
    assert any(not c.expected for c in cases), "no action"
    assert any(c.email.has_attachments for c in cases), "attachments"
    assert any(len(c.email.receivers) >= 2 for c in cases), "several receivers"
    assert any(a.due_date for c in cases for a in c.expected), "deadline"
    assert ActionType.UNCLASSIFIED in all_types


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    case = (
        '{"id": "x", "expected": [], "email": {"sender": "a@b.com", "receivers": [],'
        ' "subject": "s", "date": "2026-11-23T09:00:00+00:00", "body": "b"}}'
    )
    path = tmp_path / "dup.json"
    path.write_text(f"[{case}, {case}]", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(path)


def test_rejects_invalid_case(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('[{"id": "x", "expected": [], "email": {"sender": "a@b.com"}}]', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_dataset(path)
