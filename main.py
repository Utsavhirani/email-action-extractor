"""Run the extractor on one sample email and print the result."""

from datetime import datetime, timedelta, timezone

from core.extractor import active_mode, extract_actions
from core.models import Email


def sample_email() -> Email:
    """A small email with a review request, a meeting request and an attachment."""
    return Email(
        sender="alex@example.com",
        receivers=["priya@example.com", "sam@example.com"],
        subject="Q4 vendor contract",
        date=datetime(2026, 11, 23, 9, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        body=(
            "Hi Priya,\n\n"
            "Please review the attached vendor contract by Friday and flag anything unusual.\n"
            "Also, could you set up a call with their legal team next week?\n\n"
            "Thanks,\nAlex"
        ),
        files=["vendor_contract_v3.pdf"],
    )


def main() -> None:
    """Print the active mode and the extracted actions."""
    email = sample_email()
    print(f"Mode: {active_mode()}")
    print(f"Email: {email.subject!r} from {email.sender}\n")
    actions = extract_actions(email)
    if not actions:
        print("No actions found.")
    for i, action in enumerate(actions, 1):
        print(f"{i}. {action.action_type.value}")
        print(f"   owner: {action.owner}   due: {action.due_date}")
        print(f"   {action.action_description}")


if __name__ == "__main__":
    main()
