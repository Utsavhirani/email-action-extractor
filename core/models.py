"""Data models shared by the extractors and the evaluator."""

from datetime import date
from enum import Enum

from pydantic import AwareDatetime, BaseModel, Field, computed_field


def normalize_address(address: str) -> str:
    """Canonical form for comparing email addresses: trimmed and lowercased."""
    return address.strip().lower()


class Email(BaseModel):
    """An incoming email. Addresses are plain, e.g. 'priya@example.com'."""

    sender: str
    receivers: list[str]
    subject: str
    date: AwareDatetime  # must carry a timezone
    body: str
    files: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_attachments(self) -> bool:
        """True if the email has at least one attached file."""
        return len(self.files) > 0

    def participants(self) -> set[str]:
        """Normalized addresses of the sender and all receivers."""
        return {normalize_address(a) for a in [self.sender, *self.receivers]}


class ActionType(str, Enum):
    """The kinds of business action an email can ask for."""

    REPLY = "REPLY"
    SHARE_DOCUMENT = "SHARE_DOCUMENT"
    REVIEW_DOCUMENT = "REVIEW_DOCUMENT"
    APPROVE = "APPROVE"
    SCHEDULE_MEETING = "SCHEDULE_MEETING"
    PAY_INVOICE = "PAY_INVOICE"
    COMPLETE_TASK = "COMPLETE_TASK"
    FOLLOW_UP = "FOLLOW_UP"
    PROVIDE_INFORMATION = "PROVIDE_INFORMATION"
    UNCLASSIFIED = "UNCLASSIFIED"


class ExtractedAction(BaseModel):
    """One action point. `action_description` is for humans and is never graded."""

    action_type: ActionType
    owner: str | None = None  # an email address of a participant
    due_date: date | None = None
    action_description: str


class ExtractionResult(BaseModel):
    """The LLM's structured output: all action points in one email."""

    actions: list[ExtractedAction]
