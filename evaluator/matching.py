"""Pair predicted actions with expected actions for one email."""

from dataclasses import dataclass, field

from core.models import ExtractedAction, normalize_address


@dataclass
class MatchResult:
    """Outcome of matching. Pairs are (predicted, expected), in expected-list order."""

    pairs: list[tuple[ExtractedAction, ExtractedAction]] = field(default_factory=list)
    false_positives: list[ExtractedAction] = field(default_factory=list)  # predicted, unmatched
    false_negatives: list[ExtractedAction] = field(default_factory=list)  # expected, unmatched


def _owner_key(action: ExtractedAction) -> str | None:
    """Owner in comparable form (None stays None)."""
    return normalize_address(action.owner) if action.owner else None


def match_actions(predicted: list[ExtractedAction], expected: list[ExtractedAction]) -> MatchResult:
    """Match by action_type only (strict: a wrong type never matches).

    Two passes make the result independent of list order:
      1. pair same type AND same owner;
      2. pair remaining actions with the same type.
    Within a pass, a candidate with the same due_date is preferred (tie-break only).
    """
    used: set[int] = set()  # indexes into `predicted`
    pair_for: dict[int, int] = {}  # expected index -> predicted index

    def pick(exp: ExtractedAction, same_owner: bool) -> int | None:
        candidates = [
            i
            for i, pred in enumerate(predicted)
            if i not in used
            and pred.action_type == exp.action_type
            and (not same_owner or _owner_key(pred) == _owner_key(exp))
        ]
        same_due = [i for i in candidates if predicted[i].due_date == exp.due_date]
        return (same_due or candidates or [None])[0]

    for same_owner in (True, False):
        for e_idx, exp in enumerate(expected):
            if e_idx in pair_for:
                continue
            p_idx = pick(exp, same_owner)
            if p_idx is not None:
                used.add(p_idx)
                pair_for[e_idx] = p_idx

    return MatchResult(
        pairs=[(predicted[pair_for[e]], expected[e]) for e in sorted(pair_for)],
        false_positives=[p for i, p in enumerate(predicted) if i not in used],
        false_negatives=[e for i, e in enumerate(expected) if i not in pair_for],
    )
