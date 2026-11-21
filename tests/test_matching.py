"""Tests for evaluator.matching."""

import itertools
from datetime import date

from core.models import ActionType, ExtractedAction
from evaluator.matching import MatchResult, match_actions

A, B, C = ActionType.REVIEW_DOCUMENT, ActionType.SCHEDULE_MEETING, ActionType.REPLY


def act(t: ActionType, owner: str | None = None, due: date | None = None, desc: str = "x") -> ExtractedAction:
    return ExtractedAction(action_type=t, owner=owner, due_date=due, action_description=desc)


def owners_correct(result: MatchResult) -> int:
    return sum(p.owner == e.owner for p, e in result.pairs)


def counts(result: MatchResult) -> tuple[int, int, int]:
    return len(result.pairs), len(result.false_positives), len(result.false_negatives)


def test_perfect_match() -> None:
    expected = [act(A, "p@x.com"), act(B, "s@x.com")]
    result = match_actions(list(expected), expected)
    assert counts(result) == (2, 0, 0)
    assert owners_correct(result) == 2


def test_both_empty() -> None:
    assert counts(match_actions([], [])) == (0, 0, 0)


def test_empty_prediction() -> None:
    expected = [act(A), act(B)]
    result = match_actions([], expected)
    assert counts(result) == (0, 0, 2)
    assert result.false_negatives == expected


def test_extra_prediction() -> None:
    extra = act(C)
    result = match_actions([act(A), extra], [act(A)])
    assert counts(result) == (1, 1, 0)
    assert result.false_positives == [extra]


def test_wrong_type_never_matches() -> None:
    result = match_actions([act(B, "p@x.com")], [act(A, "p@x.com")])
    assert counts(result) == (0, 1, 1)


def test_duplicate_types() -> None:
    # Two expected of the same type, three predicted of that type: 2 pairs, 1 FP.
    result = match_actions([act(A), act(A), act(A)], [act(A), act(A)])
    assert counts(result) == (2, 1, 0)
    # Fewer predicted than expected: 1 pair, 1 FN.
    assert counts(match_actions([act(A)], [act(A), act(A)])) == (1, 0, 1)


def test_reordered_lists() -> None:
    expected = [act(A, "p@x.com"), act(B, "s@x.com"), act(C, "t@x.com")]
    result = match_actions(list(reversed(expected)), expected)
    assert counts(result) == (3, 0, 0)
    assert all(p.action_type == e.action_type for p, e in result.pairs)
    assert [e for _, e in result.pairs] == expected  # pairs follow expected order


def test_owner_tie_break() -> None:
    wrong, right = act(A, "sam@x.com", desc="wrong"), act(A, "priya@x.com", desc="right")
    result = match_actions([wrong, right], [act(A, "priya@x.com")])
    assert result.pairs[0][0] is right
    assert result.false_positives == [wrong]


def test_owner_comparison_ignores_case_and_spaces() -> None:
    pred = act(A, " Priya@X.com ", desc="match me")
    other = act(A, "sam@x.com")
    result = match_actions([other, pred], [act(A, "priya@x.com")])
    assert result.pairs[0][0] is pred


def test_none_owner_matches_none_owner_first() -> None:
    no_owner = act(A, None, desc="none")
    result = match_actions([act(A, "p@x.com"), no_owner], [act(A, None)])
    assert result.pairs[0][0] is no_owner


def test_due_date_tie_break() -> None:
    d1, d2 = date(2026, 11, 27), date(2026, 11, 28)
    late, on_time = act(A, "p@x.com", d2), act(A, "p@x.com", d1)
    result = match_actions([late, on_time], [act(A, "p@x.com", d1)])
    assert result.pairs[0][0] is on_time


def test_greedy_order_dependence_example() -> None:
    # A one-pass greedy match would pair x<->y and y<->z, giving owner accuracy 0/2.
    expected = [act(A, "x@x.com"), act(A, "y@x.com")]
    predicted = [act(A, "y@x.com"), act(A, "z@x.com")]
    assert owners_correct(match_actions(predicted, expected)) == 1


def test_result_independent_of_order() -> None:
    expected = [act(A, "x@x.com"), act(A, "y@x.com"), act(B, "x@x.com")]
    predicted = [act(A, "y@x.com"), act(A, "z@x.com"), act(B, "x@x.com"), act(C)]
    results = {
        (counts(r := match_actions(list(p), list(e))), owners_correct(r))
        for p in itertools.permutations(predicted)
        for e in itertools.permutations(expected)
    }
    assert results == {((3, 1, 0), 2)}
