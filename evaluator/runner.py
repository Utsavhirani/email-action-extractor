"""Run an extractor over a dataset and build a Report.

The evaluator never imports the extractor: it gets any function Email -> list[ExtractedAction],
plus a mode label and prompt version from the caller.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from core.models import Email, ExtractedAction
from evaluator import metrics
from evaluator.dataset import GoldenCase
from evaluator.matching import match_actions

ExtractFn = Callable[[Email], list[ExtractedAction]]


class CaseResult(BaseModel):
    """Outcome for one golden case."""

    id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    field_misses: list[str] = []  # e.g. ["due_date"], one entry per wrong field per pair
    failures: list[str] = []  # readable reasons, empty when the case is perfect
    error: str | None = None
    predicted: list[ExtractedAction] = []
    expected: list[ExtractedAction] = []

    @property
    def is_perfect(self) -> bool:
        """No error, nothing missed or extra, and every graded field right."""
        return self.error is None and not self.fp and not self.fn and not self.field_misses


class Report(BaseModel):
    """Aggregate scores (micro-averaged) plus per-case details."""

    mode: str
    prompt_version: str
    timestamp: str
    num_cases: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    owner_accuracy: float | None
    due_date_accuracy: float | None
    perfect_case_rate: float
    no_action_correct: int
    no_action_total: int
    error_count: int
    cases: list[CaseResult]

    def summary(self) -> str:
        """Human-readable report: headline numbers, then each imperfect case."""

        def pct(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.1%}"

        perfect = sum(c.is_perfect for c in self.cases)
        lines = [
            f"Eval report  mode={self.mode}  prompt={self.prompt_version}  at {self.timestamp}",
            f"Cases: {self.num_cases}   errors: {self.error_count}",
            f"Precision {pct(self.precision)}   Recall {pct(self.recall)}   F1 {pct(self.f1)}"
            f"   (TP={self.tp} FP={self.fp} FN={self.fn})",
            f"Owner accuracy {pct(self.owner_accuracy)}   Due-date accuracy {pct(self.due_date_accuracy)}"
            "   (on matched pairs)",
            f"Perfect cases {perfect}/{self.num_cases} ({pct(self.perfect_case_rate)})"
            f"   No-action correct {self.no_action_correct}/{self.no_action_total}",
        ]
        failing = [c for c in self.cases if not c.is_perfect]
        if failing:
            lines.append("\nFailures:")
            for case in failing:
                lines.append(f"  {case.id}  (TP={case.tp} FP={case.fp} FN={case.fn})")
                lines.extend(f"    - {reason}" for reason in case.failures)
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        """Write the report as JSON (creating parent folders). Returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


def _describe(action: ExtractedAction) -> str:
    return f"{action.action_type.value} (owner={action.owner}, due={action.due_date})"


def evaluate_case(case: GoldenCase, extract_fn: ExtractFn) -> tuple[CaseResult, metrics.Counts]:
    """Run, match and grade one case. An exception counts every expected action as missed."""
    try:
        predicted = extract_fn(case.email)
    except Exception as exc:  # noqa: BLE001 - one bad case must not stop the run
        error = f"{type(exc).__name__}: {exc}"
        counts = metrics.Counts(fn=len(case.expected))
        result = CaseResult(
            id=case.id, fn=counts.fn, error=error, failures=[f"error: {error}"], expected=case.expected
        )
        return result, counts

    match = match_actions(predicted, case.expected)
    counts = metrics.count_match(match)
    failures: list[str] = []
    field_misses: list[str] = []

    for exp in match.false_negatives:
        failures.append(f"missed: {_describe(exp)}")
    for pred in match.false_positives:
        failures.append(f"extra: {_describe(pred)} - {pred.action_description!r}")
    for pred, exp in match.pairs:
        for name, ok in metrics.grade_fields(pred, exp).items():
            if not ok:
                field_misses.append(name)
                failures.append(
                    f"wrong {name} on {exp.action_type.value}: expected {getattr(exp, name)}, "
                    f"got {getattr(pred, name)} - {pred.action_description!r}"
                )

    result = CaseResult(
        id=case.id,
        tp=counts.tp,
        fp=counts.fp,
        fn=counts.fn,
        field_misses=field_misses,
        failures=failures,
        predicted=predicted,
        expected=case.expected,
    )
    return result, counts


def evaluate(
    dataset: list[GoldenCase],
    extract_fn: ExtractFn,
    mode: str = "unknown",
    prompt_version: str = "unknown",
) -> Report:
    """Evaluate extract_fn over every case and return a Report."""
    if not dataset:
        raise ValueError("Dataset is empty")

    results: list[CaseResult] = []
    all_counts: list[metrics.Counts] = []
    for case in dataset:
        result, counts = evaluate_case(case, extract_fn)
        results.append(result)
        all_counts.append(counts)

    summed = metrics.total(all_counts)
    no_action = [r for r in results if not r.expected]
    return Report(
        mode=mode,
        prompt_version=prompt_version,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        num_cases=len(results),
        tp=summed.tp,
        fp=summed.fp,
        fn=summed.fn,
        precision=metrics.precision(summed),
        recall=metrics.recall(summed),
        f1=metrics.f1(summed),
        owner_accuracy=metrics.field_accuracy(summed, "owner"),
        due_date_accuracy=metrics.field_accuracy(summed, "due_date"),
        perfect_case_rate=sum(r.is_perfect for r in results) / len(results),
        no_action_correct=sum(r.error is None and not r.predicted for r in no_action),
        no_action_total=len(no_action),
        error_count=sum(r.error is not None for r in results),
        cases=results,
    )
