"""Offline keyword-based extractor. Deliberately dumb: a baseline the LLM should beat."""

import re

from core.models import ActionType, Email, ExtractedAction, normalize_address

# (pattern, action type). Word boundaries stop "call" matching "recall".
_KEYWORD_RULES: list[tuple[str, ActionType]] = [
    (r"\binvoice\b", ActionType.PAY_INVOICE),
    (r"\b(meeting|call)\b", ActionType.SCHEDULE_MEETING),
    (r"\b(approve|approval)\b", ActionType.APPROVE),
]
_REVIEW_PATTERN = r"\breview\b"


def extract_actions_stub(email: Email) -> list[ExtractedAction]:
    """Match keywords in subject + body. Owner = first receiver, never a due date."""
    text = f"{email.subject}\n{email.body}".lower()
    owner = normalize_address(email.receivers[0]) if email.receivers else None

    found: list[tuple[ActionType, str]] = []
    for pattern, action_type in _KEYWORD_RULES:
        match = re.search(pattern, text)
        if match:
            found.append((action_type, match.group(0)))
    if email.has_attachments and re.search(_REVIEW_PATTERN, text):
        found.append((ActionType.REVIEW_DOCUMENT, "review + attachment"))

    return [
        ExtractedAction(
            action_type=action_type,
            owner=owner,
            due_date=None,
            action_description=f"[stub] matched {reason!r}",
        )
        for action_type, reason in found
    ]
