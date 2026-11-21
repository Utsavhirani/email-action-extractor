"""Precision / recall / F1 and per-field accuracy, micro-averaged over all cases.

Zero-denominator choices (documented here, tested in tests/test_metrics.py):
- precision = 1.0 when nothing was predicted (TP+FP == 0): no false claims were made.
- recall    = 1.0 when nothing was expected (TP+FN == 0): there was nothing to miss.
- F1        = 0.0 when precision + recall == 0.
  So an extractor that predicts nothing gets P=1.0, R=0.0, F1=0.0, which still fails loudly.
- field accuracy = None when there are no matched pairs (not measurable, shown as "n/a").
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from core.models import ExtractedAction, normalize_address
from evaluator.matching import MatchResult

# Fields graded on matched pairs (exact match). action_type is already enforced by matching.
GRADED_FIELDS: tuple[str, ...] = ("owner", "due_date")
# Never graded: free text for humans, and wording varies legitimately.
UNGRADED_FIELDS: tuple[str, ...] = ("action_description",)


def _empty_field_counts() -> dict[str, int]:
    return {name: 0 for name in GRADED_FIELDS}


@dataclass
class Counts:
    """Raw tallies. Add them across cases, then compute ratios (= micro-averaging)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    field_correct: dict[str, int] = field(default_factory=_empty_field_counts)

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            field_correct={f: self.field_correct[f] + other.field_correct[f] for f in GRADED_FIELDS},
        )


def grade_fields(predicted: ExtractedAction, expected: ExtractedAction) -> dict[str, bool]:
    """Exact-match each graded field of a matched pair (owners compared normalized)."""
    results: dict[str, bool] = {}
    for name in GRADED_FIELDS:
        got, want = getattr(predicted, name), getattr(expected, name)
        if name == "owner":
            got = normalize_address(got) if got else None
            want = normalize_address(want) if want else None
        results[name] = got == want
    return results


def count_match(match: MatchResult) -> Counts:
    """Tally one case's MatchResult."""
    counts = Counts(tp=len(match.pairs), fp=len(match.false_positives), fn=len(match.false_negatives))
    for predicted, expected in match.pairs:
        for name, ok in grade_fields(predicted, expected).items():
            counts.field_correct[name] += ok
    return counts


def total(all_counts: Iterable[Counts]) -> Counts:
    """Sum tallies over all cases."""
    return sum(all_counts, Counts())


def precision(c: Counts) -> float:
    """TP / (TP + FP); 1.0 if nothing was predicted."""
    return c.tp / (c.tp + c.fp) if c.tp + c.fp else 1.0


def recall(c: Counts) -> float:
    """TP / (TP + FN); 1.0 if nothing was expected."""
    return c.tp / (c.tp + c.fn) if c.tp + c.fn else 1.0


def f1(c: Counts) -> float:
    """Harmonic mean of precision and recall; 0.0 if both are 0."""
    p, r = precision(c), recall(c)
    return 2 * p * r / (p + r) if p + r else 0.0


def field_accuracy(c: Counts, name: str) -> float | None:
    """Share of matched pairs where `name` was exactly right; None if there are no pairs."""
    if name not in GRADED_FIELDS:
        raise ValueError(f"{name!r} is not a graded field; graded fields are {GRADED_FIELDS}")
    return c.field_correct[name] / c.tp if c.tp else None
