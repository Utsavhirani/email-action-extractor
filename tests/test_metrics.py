"""Tests for evaluator.metrics."""

from datetime import date

import pytest

from core.models import ActionType, ExtractedAction
from evaluator.matching import match_actions
from evaluator.metrics import (
    GRADED_FIELDS,
    Counts,
    count_match,
    f1,
    field_accuracy,
    grade_fields,
    precision,
    recall,
    total,
)

PRIYA = "priya@example.com"


def act(t: ActionType, owner: str | None = PRIYA, due: date | None = None, desc: str = "x") -> ExtractedAction:
    return ExtractedAction(action_type=t, owner=owner, due_date=due, action_description=desc)


def test_worked_example() -> None:
    expected = [
        act(ActionType.REVIEW_DOCUMENT, due=date(2026, 11, 27)),
        act(ActionType.SCHEDULE_MEETING, due=None),
    ]
    predicted = [
        act(ActionType.REVIEW_DOCUMENT, due=date(2026, 11, 28)),
        act(ActionType.REPLY, due=None),
    ]
    c = count_match(match_actions(predicted, expected))

    assert (c.tp, c.fp, c.fn) == (1, 1, 1)
    assert precision(c) == 0.5
    assert recall(c) == 0.5
    assert f1(c) == 0.5
    assert field_accuracy(c, "owner") == 1.0  # 1/1
    assert field_accuracy(c, "due_date") == 0.0  # 0/1


def test_description_is_not_graded() -> None:
    assert "action_description" not in GRADED_FIELDS
    pred = act(ActionType.APPROVE, desc="totally different words")
    exp = act(ActionType.APPROVE, desc="approve the budget")
    assert all(grade_fields(pred, exp).values())
    with pytest.raises(ValueError):
        field_accuracy(Counts(tp=1), "action_description")


def test_owner_graded_normalized() -> None:
    assert grade_fields(act(ActionType.APPROVE, owner=" Priya@Example.COM"), act(ActionType.APPROVE))["owner"]
    assert grade_fields(act(ActionType.APPROVE, owner=None), act(ActionType.APPROVE, owner=None))["owner"]
    assert not grade_fields(act(ActionType.APPROVE, owner=None), act(ActionType.APPROVE))["owner"]


def test_micro_average_sums_counts_before_dividing() -> None:
    # Case 1: perfect (TP=1). Case 2: three false positives.
    case1 = count_match(match_actions([act(ActionType.APPROVE)], [act(ActionType.APPROVE)]))
    case2 = count_match(match_actions([act(ActionType.REPLY)] * 3, []))
    summed = total([case1, case2])
    assert (summed.tp, summed.fp, summed.fn) == (1, 3, 0)
    assert precision(summed) == 0.25  # micro: 1/4, not the macro mean (1.0 + 0.0) / 2 = 0.5
    assert summed.field_correct == {"owner": 1, "due_date": 1}


def test_total_of_nothing() -> None:
    assert total([]) == Counts()


def test_zero_denominators_nothing_expected_nothing_predicted() -> None:
    c = Counts()
    assert (precision(c), recall(c), f1(c)) == (1.0, 1.0, 1.0)
    assert field_accuracy(c, "owner") is None


def test_zero_denominators_nothing_predicted() -> None:
    c = Counts(fn=2)
    assert (precision(c), recall(c), f1(c)) == (1.0, 0.0, 0.0)


def test_zero_denominators_all_wrong() -> None:
    c = Counts(fp=2, fn=2)
    assert (precision(c), recall(c), f1(c)) == (0.0, 0.0, 0.0)
    assert field_accuracy(c, "due_date") is None
