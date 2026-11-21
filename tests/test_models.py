"""Tests for core.models."""

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from core.models import ActionType, Email, ExtractedAction, ExtractionResult

UTC_DATE = datetime(2026, 11, 23, 9, 0, tzinfo=timezone.utc)


def make_email(**overrides: object) -> Email:
    """Build a valid Email, with optional field overrides."""
    fields: dict[str, object] = {
        "sender": "alex@example.com",
        "receivers": ["priya@example.com"],
        "subject": "Contract review",
        "date": UTC_DATE,
        "body": "Please review the attached contract by Friday.",
    }
    fields.update(overrides)
    return Email(**fields)  # type: ignore[arg-type]


def test_email_valid_defaults() -> None:
    email = make_email()
    assert email.files == []
    assert email.has_attachments is False


def test_email_has_attachments() -> None:
    assert make_email(files=["contract.pdf"]).has_attachments is True


def test_email_date_parsed_from_iso_string() -> None:
    email = make_email(date="2026-11-23T09:00:00+05:30")
    assert email.date.utcoffset() is not None


def test_email_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError):
        make_email(date=datetime(2026, 11, 23, 9, 0))


def test_email_rejects_missing_field() -> None:
    with pytest.raises(ValidationError):
        Email(sender="a@example.com", receivers=[], subject="x", date=UTC_DATE)  # type: ignore[call-arg]


def test_has_attachments_in_serialized_output() -> None:
    assert make_email(files=["a.pdf"]).model_dump()["has_attachments"] is True


def test_extracted_action_valid() -> None:
    action = ExtractedAction(
        action_type="REVIEW_DOCUMENT",  # type: ignore[arg-type]
        owner="priya@example.com",
        due_date="2026-11-27",  # type: ignore[arg-type]
        action_description="Review the contract",
    )
    assert action.action_type is ActionType.REVIEW_DOCUMENT
    assert action.due_date == date(2026, 11, 27)


def test_extracted_action_optional_fields_default_to_none() -> None:
    action = ExtractedAction(action_type=ActionType.REPLY, action_description="Reply")
    assert action.owner is None
    assert action.due_date is None


def test_extracted_action_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        ExtractedAction(action_type="DANCE", action_description="x")  # type: ignore[arg-type]


def test_extracted_action_rejects_bad_date() -> None:
    with pytest.raises(ValidationError):
        ExtractedAction(
            action_type=ActionType.REPLY,
            due_date="next friday",  # type: ignore[arg-type]
            action_description="x",
        )


def test_extraction_result_from_json() -> None:
    raw = '{"actions": [{"action_type": "PAY_INVOICE", "owner": null, "due_date": null, "action_description": "Pay it"}]}'
    result = ExtractionResult.model_validate_json(raw)
    assert result.actions[0].action_type is ActionType.PAY_INVOICE


def test_extraction_result_empty() -> None:
    assert ExtractionResult(actions=[]).actions == []
