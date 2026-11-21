"""Tests for the extractors and the USE_LLM switch. Never touches the network."""

from datetime import date, datetime, timezone
import json
from typing import Any

import httpx
import pytest

from core import config, extractor, llm_extractor
from core.models import ActionType, Email, ExtractedAction
from core.prompts import ACTION_DEFINITIONS, build_user_prompt
from core.stub_extractor import extract_actions_stub


def make_email(body: str, subject: str = "Hello", files: list[str] | None = None) -> Email:
    """A Monday-morning email from alex to priya and sam."""
    return Email(
        sender="alex@example.com",
        receivers=["Priya@Example.com", "sam@example.com"],
        subject=subject,
        date=datetime(2026, 11, 23, 9, 0, tzinfo=timezone.utc),
        body=body,
        files=files or [],
    )


def action(t: ActionType, owner: str | None = None) -> ExtractedAction:
    return ExtractedAction(action_type=t, owner=owner, action_description="x")


# ---------- stub extractor ----------


def types_of(actions: list[ExtractedAction]) -> list[ActionType]:
    return [a.action_type for a in actions]


def test_stub_invoice() -> None:
    result = extract_actions_stub(make_email("Please pay the attached invoice."))
    assert types_of(result) == [ActionType.PAY_INVOICE]
    assert result[0].owner == "priya@example.com"  # first receiver, normalized
    assert result[0].due_date is None


def test_stub_meeting_and_call() -> None:
    assert types_of(extract_actions_stub(make_email("Can we have a call?"))) == [
        ActionType.SCHEDULE_MEETING
    ]


def test_stub_word_boundaries() -> None:
    assert extract_actions_stub(make_email("I recall the invoices were fine.")) == []


def test_stub_review_needs_attachment() -> None:
    assert extract_actions_stub(make_email("Please review this.")) == []
    with_file = extract_actions_stub(make_email("Please review this.", files=["a.pdf"]))
    assert types_of(with_file) == [ActionType.REVIEW_DOCUMENT]


def test_stub_multiple_actions() -> None:
    email = make_email("Review the invoice and book a meeting.", files=["inv.pdf"])
    assert set(types_of(extract_actions_stub(email))) == {
        ActionType.PAY_INVOICE,
        ActionType.SCHEDULE_MEETING,
        ActionType.REVIEW_DOCUMENT,
    }


def test_stub_no_action() -> None:
    assert extract_actions_stub(make_email("FYI, the office is closed Monday.")) == []


def test_stub_is_deterministic() -> None:
    email = make_email("Pay the invoice, then schedule a meeting.")
    assert extract_actions_stub(email) == extract_actions_stub(email)


# ---------- config + switch ----------


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False), ("true", False), ("", False)])
def test_use_llm_values(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.setenv("USE_LLM", value)
    assert config.use_llm() is expected


def test_use_llm_unset_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USE_LLM", raising=False)
    assert config.use_llm() is False


def test_llm_model_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert config.llm_model() == config.DEFAULT_LLM_MODEL
    monkeypatch.setenv("LLM_MODEL", "vendor/test:free")
    assert config.llm_model() == "vendor/test:free"


def test_missing_api_key_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        config.openrouter_api_key()


def test_call_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_CALL_DELAY_SECONDS", raising=False)
    assert config.llm_call_delay_seconds() == 0.0
    monkeypatch.setenv("LLM_CALL_DELAY_SECONDS", "4.5")
    assert config.llm_call_delay_seconds() == 4.5
    monkeypatch.setenv("LLM_CALL_DELAY_SECONDS", "soon")
    with pytest.raises(ValueError):
        config.llm_call_delay_seconds()


def test_env_example_lists_every_setting() -> None:
    """Every variable config.py reads must appear in the .env.example template."""
    import re
    from pathlib import Path

    read_vars = set(re.findall(r'os\.environ\.get\("([A-Z_]+)"', Path(config.__file__).read_text()))
    template = (config.ENV_FILE.parent / ".env.example").read_text()
    template_vars = set(re.findall(r"^([A-Z_]+)=", template, flags=re.MULTILINE))
    assert read_vars == template_vars


def test_switch_off_uses_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USE_LLM", raising=False)

    def boom(email: Email) -> list[ExtractedAction]:
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(llm_extractor, "extract_actions_llm", boom)
    result = extractor.extract_actions(make_email("Please pay the invoice."))
    assert types_of(result) == [ActionType.PAY_INVOICE]
    assert extractor.active_mode() == "stub"


def test_switch_on_uses_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_LLM", "1")
    monkeypatch.setenv("LLM_MODEL", "vendor/test:free")
    fake_result = [action(ActionType.APPROVE)]
    monkeypatch.setattr(llm_extractor, "extract_actions_llm", lambda email: fake_result)
    assert extractor.extract_actions(make_email("Please pay the invoice.")) == fake_result
    assert extractor.active_mode() == "vendor/test:free"


# ---------- LLM extractor (mocked HTTP) ----------


def chat_reply(text: str, model: str = "main/model:free") -> dict[str, Any]:
    """A minimal successful chat/completions response body."""
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }


class FakeOpenRouter:
    """Fake HTTP transport. Script items: str (reply text), dict (200 body),
    int (HTTP error), (int, message) tuple, httpx.Response, or Exception."""

    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, httpx.Response):
            return item
        if isinstance(item, int):
            item = (item, f"err {item}")
        if isinstance(item, tuple):
            status, message = item
            return httpx.Response(status, json={"error": {"code": status, "message": message}})
        if isinstance(item, dict):
            return httpx.Response(200, json=item)
        return httpx.Response(200, json=chat_reply(item))

    def body(self, i: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[i].content)


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch):
    """Install a FakeOpenRouter transport; no real network, no real sleeping."""

    def install(script: list[Any]) -> FakeOpenRouter:
        fake = FakeOpenRouter(script)
        client = httpx.Client(transport=httpx.MockTransport(fake.handler))
        monkeypatch.setattr(llm_extractor, "_http", client)
        monkeypatch.setattr(llm_extractor.time, "sleep", lambda s: None)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "main/model:free")
        monkeypatch.setenv("LLM_FALLBACK_MODELS", "backup/model:free")
        monkeypatch.delenv("LLM_CALL_DELAY_SECONDS", raising=False)
        return fake

    return install


EMPTY = '{"actions": []}'


def test_llm_extractor_parses_and_cleans(fake_api) -> None:
    raw = """{"actions": [
      {"action_type": "REVIEW_DOCUMENT", "owner": " PRIYA@example.com ", "due_date": "2026-11-27", "action_description": "Review"},
      {"action_type": "SCHEDULE_MEETING", "owner": "stranger@other.com", "due_date": null, "action_description": "Meet"}
    ]}"""
    fake_api([raw])
    result = llm_extractor.extract_actions_llm(make_email("Review it by Friday.", files=["a.pdf"]))

    assert result[0].owner == "priya@example.com"  # normalized, kept
    assert result[0].due_date == date(2026, 11, 27)
    assert result[1].owner is None  # not a participant -> dropped, not guessed


def test_request_shape(fake_api) -> None:
    fake = fake_api([EMPTY])
    llm_extractor.extract_actions_llm(make_email("FYI only."))
    request = fake.requests[0]
    assert str(request.url) == llm_extractor.API_URL
    assert request.headers["authorization"] == "Bearer test-key"
    assert "test-key" not in str(request.url)

    body = fake.body()
    assert body["model"] == "main/model:free"
    assert body["models"] == ["main/model:free", "backup/model:free"]  # server-side fallback
    assert body["provider"] == {"require_parameters": True}
    assert body["response_format"]["type"] == "json_schema"
    schema_text = json.dumps(body["response_format"]["json_schema"]["schema"])
    assert "$defs" not in schema_text and "$ref" not in schema_text
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "FYI only." in body["messages"][1]["content"]


def test_llm_extractor_empty_list(fake_api) -> None:
    fake_api([EMPTY])
    assert llm_extractor.extract_actions_llm(make_email("FYI only.")) == []


def test_fenced_json_is_accepted(fake_api) -> None:
    fake_api(["```json\n" + EMPTY + "\n```"])
    assert llm_extractor.extract_actions_llm(make_email("FYI only.")) == []


def test_overloaded_is_retried_once(fake_api) -> None:
    fake = fake_api([503, EMPTY])
    assert llm_extractor.extract_actions_llm(make_email("FYI only.")) == []
    assert len(fake.requests) == 2


def test_gives_up_after_one_retry(fake_api) -> None:
    fake = fake_api([503, 502])
    with pytest.raises(llm_extractor.LLMError, match="HTTP 502"):
        llm_extractor.extract_actions_llm(make_email("FYI only."))
    assert len(fake.requests) == 2  # quota-friendly: at most 2 requests per email


def test_timeout_is_retried(fake_api) -> None:
    fake = fake_api([httpx.ReadTimeout("slow"), EMPTY])
    assert llm_extractor.extract_actions_llm(make_email("FYI only.")) == []
    assert len(fake.requests) == 2


def test_daily_limit_not_retried(fake_api) -> None:
    message = "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day"
    fake = fake_api([(429, message)])
    with pytest.raises(llm_extractor.LLMError, match="daily limit") as info:
        llm_extractor.extract_actions_llm(make_email("FYI only."))
    assert info.value.quota_exhausted
    assert len(fake.requests) == 1


def busy_response(retry_after: str) -> httpx.Response:
    return httpx.Response(
        429, headers={"Retry-After": retry_after}, json={"error": {"code": 429, "message": "slow down"}}
    )


def test_long_retry_after_fails_fast(fake_api) -> None:
    fake = fake_api([busy_response("600")])
    with pytest.raises(llm_extractor.LLMError, match="HTTP 429"):
        llm_extractor.extract_actions_llm(make_email("FYI only."))
    assert len(fake.requests) == 1


def test_short_retry_after_is_honoured(fake_api, monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    fake_api([busy_response("5"), EMPTY])
    monkeypatch.setattr(llm_extractor.time, "sleep", sleeps.append)
    llm_extractor.extract_actions_llm(make_email("FYI only."))
    assert sleeps == [5.0]


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
def test_fatal_errors_stop_immediately(fake_api, status: int) -> None:
    fake = fake_api([status])
    with pytest.raises(llm_extractor.LLMError) as info:
        llm_extractor.extract_actions_llm(make_email("FYI only."))
    assert info.value.status == status
    assert len(fake.requests) == 1


def test_error_inside_200_response(fake_api) -> None:
    fake = fake_api([{"error": {"code": 502, "message": "Provider returned error"}}, EMPTY])
    assert llm_extractor.extract_actions_llm(make_email("FYI only.")) == []
    assert len(fake.requests) == 2  # upstream 502 is transient -> retried


def test_malformed_output_is_retried(fake_api) -> None:
    fake = fake_api(['{"actions": [{"action_type": "NOPE"}]}', EMPTY])
    assert llm_extractor.extract_actions_llm(make_email("FYI only.")) == []
    assert len(fake.requests) == 2


def test_empty_choices() -> None:
    with pytest.raises(llm_extractor.LLMError, match="no choices"):
        llm_extractor.response_text({"choices": []})


def test_inline_refs() -> None:
    schema = {
        "$defs": {"A": {"type": "string"}},
        "properties": {"x": {"$ref": "#/$defs/A"}, "y": {"anyOf": [{"$ref": "#/$defs/A"}, {"type": "null"}]}},
    }
    assert llm_extractor.inline_refs(schema) == {
        "properties": {"x": {"type": "string"}, "y": {"anyOf": [{"type": "string"}, {"type": "null"}]}}
    }


def test_retry_backoff_schedule() -> None:
    attempts: list[int] = []
    sleeps: list[float] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise llm_extractor.LLMError("busy", status=503)
        return "ok"

    assert llm_extractor.call_with_retry(flaky, max_retries=2, base_delay=1.0, sleep=sleeps.append) == "ok"
    assert sleeps == [1.0, 2.0]  # exponential backoff


# ---------- settings ----------


def test_llm_models_order_and_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "a")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "b, a ,c")
    assert config.llm_models() == ["a", "b", "c"]
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "none")
    assert config.llm_models() == ["a"]
    monkeypatch.delenv("LLM_FALLBACK_MODELS")
    assert config.llm_models()[1:] == config.DEFAULT_LLM_FALLBACK_MODELS.split(",")


def test_llm_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
    assert config.llm_timeout_seconds() == config.DEFAULT_LLM_TIMEOUT_SECONDS
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "15")
    assert config.llm_timeout_seconds() == 15.0
    for bad in ["0", "-3", "soon"]:
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", bad)
        with pytest.raises(ValueError):
            config.llm_timeout_seconds()


# ---------- prompt ----------


def test_every_action_type_has_definition() -> None:
    assert set(ACTION_DEFINITIONS) == set(ActionType)


def test_user_prompt_contains_email_fields() -> None:
    prompt = build_user_prompt(make_email("Body text here", subject="Subj", files=["f.pdf"]))
    for expected in ["alex@example.com", "sam@example.com", "Subj", "f.pdf", "Body text here", "Monday"]:
        assert expected in prompt
