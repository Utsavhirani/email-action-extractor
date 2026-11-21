"""Prompt text for the LLM extractor."""

from core.models import ActionType, Email

# Bump this whenever the prompt changes, so eval reports can be compared fairly.
PROMPT_VERSION = "v2"

ACTION_DEFINITIONS: dict[ActionType, str] = {
    ActionType.REPLY: "Write back to the sender with an answer or acknowledgement.",
    ActionType.SHARE_DOCUMENT: "Send or give access to a file or document someone asked for.",
    ActionType.REVIEW_DOCUMENT: "Read a document and give feedback or corrections.",
    ActionType.APPROVE: "Give a formal yes/no sign-off on a request, budget, plan or document.",
    ActionType.SCHEDULE_MEETING: "Arrange or book a meeting, call or appointment.",
    ActionType.PAY_INVOICE: "Pay an invoice or bill, or arrange its payment.",
    ActionType.COMPLETE_TASK: "Finish a specific piece of assigned work (e.g. a training, updating a spreadsheet, fixing a bug).",
    ActionType.FOLLOW_UP: "Chase or check back on something later (a pending item, person or decision).",
    ActionType.PROVIDE_INFORMATION: "Supply specific facts, numbers or details that were asked for.",
    ActionType.UNCLASSIFIED: "A real request that fits none of the other types (e.g. a personal favour, a social or charity request).",
}

SYSTEM_INSTRUCTION = """You extract business action points from emails.

Rules:
- List EVERY action point the email asks for. Split compound requests into separate actions, one action type each.
- Do not list the same request twice. Do not infer actions that were not asked for.
- Sending back comments, feedback or corrections on a document is part of REVIEW_DOCUMENT, not a separate REPLY ("look over the draft and let me know your thoughts" is ONE action). Use REPLY only when the email asks for an answer or acknowledgement that is not just the result of another action you listed.
- If the email asks for nothing (FYI, newsletter, announcement), return an empty list.
- Use UNCLASSIFIED when an action exists but no other type fits.
- Resolve relative dates ("by Friday", "tomorrow", "end of next week") against the email's sent date. Output due_date as an ISO date (YYYY-MM-DD), or null if no deadline is given.
- owner is the email address of the person who must do the action. It MUST be one of the email's participants (sender or receivers). If it's unclear who, use null.
- action_description is a short, human-readable summary of the action.

Action types:
{definitions}
"""


def build_system_instruction() -> str:
    """System prompt: the rules plus a one-line definition of each action type."""
    definitions = "\n".join(f"- {t.value}: {d}" for t, d in ACTION_DEFINITIONS.items())
    return SYSTEM_INSTRUCTION.format(definitions=definitions)


def build_user_prompt(email: Email) -> str:
    """The email itself, laid out for the model. The weekday helps resolve relative dates."""
    files = ", ".join(email.files) if email.files else "(none)"
    return (
        f"From: {email.sender}\n"
        f"To: {', '.join(email.receivers)}\n"
        f"Sent: {email.date.isoformat()} ({email.date:%A})\n"
        f"Subject: {email.subject}\n"
        f"Attachments: {files}\n"
        f"\n{email.body}"
    )
