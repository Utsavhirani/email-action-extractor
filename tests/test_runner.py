"""Tests for evaluator.runner, using fake extractors over the real golden dataset."""

import json
from pathlib import Path

import pytest

from core.models import ActionType, Email, ExtractedAction
from evaluator.dataset import GoldenCase, load_dataset
from evaluator.runner import Report, evaluate


@pytest.fixture(scope="module")
def dataset() -> list[GoldenCase]:
    return load_dataset()


@pytest.fixture(scope="module")
def answers(dataset: list[GoldenCase]) -> dict[str, list[ExtractedAction]]:
    """Expected actions keyed by subject (subjects are unique in golden.json)."""
    by_subject = {c.email.subject: c.expected for c in dataset}
    assert len(by_subject) == len(dataset), "subjects must be unique for this lookup"
    return by_subject


def test_perfect_fn(dataset: list[GoldenCase], answers: dict[str, list[ExtractedAction]]) -> None:
    def perfect_fn(email: Email) -> list[ExtractedAction]:
        return list(answers[email.subject])

    report = evaluate(dataset, perfect_fn, mode="fake-perfect", prompt_version="test")
    assert (report.precision, report.recall, report.f1) == (1.0, 1.0, 1.0)
    assert report.owner_accuracy == 1.0
    assert report.due_date_accuracy == 1.0
    assert report.perfect_case_rate == 1.0
    assert report.no_action_correct == report.no_action_total == 2
    assert report.error_count == 0
    assert "Failures:" not in report.summary()


def test_empty_fn(dataset: list[GoldenCase]) -> None:
    report = evaluate(dataset, lambda email: [])
    assert report.recall == 0.0
    assert report.f1 == 0.0
    assert report.tp == 0
    assert report.owner_accuracy is None  # no pairs -> not measurable
    assert report.no_action_correct == 2  # the two no-action cases are right
    assert "n/a" in report.summary()


def test_extra_fn(dataset: list[GoldenCase], answers: dict[str, list[ExtractedAction]]) -> None:
    junk = ExtractedAction(action_type=ActionType.UNCLASSIFIED, action_description="made-up junk")

    def extra_fn(email: Email) -> list[ExtractedAction]:
        return [*answers[email.subject], junk]

    report = evaluate(dataset, extra_fn)
    assert report.recall == 1.0
    assert report.precision < 1.0
    assert report.fp == len(dataset)
    assert report.no_action_correct == 0
    assert "made-up junk" in report.summary()  # predicted description shown in failures


def test_crashing_fn(dataset: list[GoldenCase], answers: dict[str, list[ExtractedAction]]) -> None:
    crash_subject = dataset[0].email.subject

    def crashing_fn(email: Email) -> list[ExtractedAction]:
        if email.subject == crash_subject:
            raise RuntimeError("model exploded")
        return list(answers[email.subject])

    report = evaluate(dataset, crashing_fn)
    assert report.error_count == 1
    assert report.num_cases == len(dataset)  # run continued
    crashed = report.cases[0]
    assert crashed.error == "RuntimeError: model exploded"
    assert crashed.fn == len(dataset[0].expected)  # errored case counts as missed
    assert report.recall < 1.0
    assert report.precision == 1.0
    assert all(c.is_perfect for c in report.cases[1:])


def test_wrong_due_date_reported(dataset: list[GoldenCase], answers: dict[str, list[ExtractedAction]]) -> None:
    case = next(c for c in dataset if c.expected and c.expected[0].due_date)

    def off_by_one(email: Email) -> list[ExtractedAction]:
        actions = list(answers[email.subject])
        if email.subject == case.email.subject:
            first = actions[0]
            actions[0] = first.model_copy(
                update={"due_date": first.due_date.replace(day=1), "action_description": "late one"}
            )
        return actions

    report = evaluate([case], off_by_one)
    result = report.cases[0]
    assert result.field_misses == ["due_date"]
    assert not result.is_perfect
    assert "wrong due_date" in result.failures[0] and "late one" in result.failures[0]
    assert report.due_date_accuracy is not None and report.due_date_accuracy < 1.0


def test_empty_dataset_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate([], lambda email: [])


def test_save_round_trip(tmp_path: Path, dataset: list[GoldenCase]) -> None:
    report = evaluate(dataset, lambda email: [], mode="stub", prompt_version="v1")
    path = report.save(tmp_path / "nested" / "report.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["mode"] == "stub"
    assert data["prompt_version"] == "v1"
    assert data["timestamp"]
    assert len(data["cases"]) == len(dataset)
    assert Report.model_validate(data) == report


def test_evaluator_does_not_import_extractor() -> None:
    forbidden = ("core.extractor", "core.llm_extractor", "core.stub_extractor", "core import config")
    for source in Path("evaluator").glob("*.py"):
        text = source.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text, f"{source} mentions {name}"
