"""LLM extractor: plain HTTPS calls to OpenRouter's free models, no SDK.

One POST to the OpenAI-style chat-completions endpoint per email. The request carries:
- a `models` list, so OpenRouter itself falls back to the next model when one is busy
  (one HTTP request = one hit on the free daily quota, however many models it tries);
- a JSON-schema `response_format` plus `require_parameters`, so only backends that enforce
  the schema are used. The reply is still validated with Pydantic.

The free tier is small (about 50 requests/day, 20/min), so retries are kept to a minimum and
the daily-limit error is reported straight away instead of being retried.
"""

import copy
import json
import logging
import re
import time
from typing import Any, Callable, TypeVar

import httpx
from pydantic import ValidationError

from core import config
from core.models import Email, ExtractedAction, ExtractionResult, normalize_address
from core.prompts import build_system_instruction, build_user_prompt

T = TypeVar("T")
log = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
APP_TITLE = "email-action-extractor"  # shown in the OpenRouter dashboard; nothing personal

MAX_RETRIES = 1  # every attempt counts against the free daily quota
BASE_BACKOFF_SECONDS = 2.0
MAX_WAIT_SECONDS = 30.0  # never honour a Retry-After longer than this; fail instead
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_http: httpx.Client | None = None


class LLMError(RuntimeError):
    """A failed LLM call. `status` is the HTTP code, or None for network/timeout/bad output."""

    def __init__(
        self,
        message: str,
        status: int | None = None,
        wait: float | None = None,
        quota_exhausted: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.wait = wait  # server-suggested pause before retrying, if any
        self.quota_exhausted = quota_exhausted  # daily free limit: retrying can't help

    @property
    def retryable(self) -> bool:
        """Transient: worth repeating the same request once after a pause."""
        if self.quota_exhausted:
            return False
        return self.status is None or self.status in RETRYABLE_STATUS


def _get_http() -> httpx.Client:
    """Create the HTTP client once, on first use."""
    global _http
    if _http is None:
        _http = httpx.Client()
    return _http


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace every {"$ref": "#/$defs/X"} with the definition itself; drop $defs.

    Some backends reject $ref, and ours has no recursive types, so inlining is safe.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def output_schema() -> dict[str, Any]:
    return inline_refs(ExtractionResult.model_json_schema())


def build_system_prompt() -> str:
    """The shared rules plus the exact JSON shape, for models that follow schemas loosely."""
    return (
        build_system_instruction()
        + "\nReply with ONLY a JSON object matching this JSON Schema, no other text:\n"
        + json.dumps(output_schema())
    )


def build_request_body(email: Email) -> dict[str, Any]:
    """chat/completions payload with model fallback and a JSON-schema response format."""
    models = config.llm_models()
    return {
        "model": models[0],
        "models": models,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(email)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "ExtractionResult", "strict": False, "schema": output_schema()},
        },
        "provider": {"require_parameters": True},
        "temperature": 0,
    }


def _error_message(payload: Any, fallback: str) -> str:
    """OpenRouter's error message (plus upstream detail if present), else the raw text."""
    try:
        err = payload["error"]
        message = str(err.get("message", ""))
        raw = (err.get("metadata") or {}).get("raw")
        return f"{message} ({raw})" if raw else message or fallback
    except Exception:
        return fallback


def _wait_hint(response: httpx.Response) -> float | None:
    """Seconds to wait from Retry-After or X-RateLimit-Reset (epoch ms), if given."""
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    reset = response.headers.get("x-ratelimit-reset")
    if reset:
        try:
            return max(0.0, float(reset) / 1000 - time.time())
        except ValueError:
            pass
    return None


def _is_daily_limit(message: str) -> bool:
    return bool(re.search(r"per-day|free-models-per-day|daily", message, re.IGNORECASE))


def post_chat(body: dict[str, Any]) -> dict[str, Any]:
    """One HTTP call. Returns the parsed JSON body, or raises LLMError."""
    timeout = config.llm_timeout_seconds()
    try:
        response = _get_http().post(
            API_URL,
            headers={
                "Authorization": f"Bearer {config.openrouter_api_key()}",
                "X-Title": APP_TITLE,
            },
            json=body,
            timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        raise LLMError(f"no response within {timeout:.0f}s") from exc
    except httpx.TransportError as exc:
        raise LLMError(f"network error: {exc}") from exc

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if response.status_code != 200:
        message = _error_message(payload, response.text[:300])
        if response.status_code == 429 and _is_daily_limit(message):
            raise LLMError(
                f"OpenRouter free daily limit reached (resets 00:00 UTC): {message}",
                status=429,
                quota_exhausted=True,
            )
        raise LLMError(
            f"HTTP {response.status_code}: {message}",
            status=response.status_code,
            wait=_wait_hint(response),
        )
    if not isinstance(payload, dict):
        raise LLMError(f"response was not JSON: {response.text[:200]}")
    if "error" in payload:  # OpenRouter sometimes reports upstream failures inside a 200
        err = payload["error"]
        code = err.get("code") if isinstance(err, dict) else None
        raise LLMError(
            f"upstream error: {_error_message(payload, str(err))}",
            status=code if isinstance(code, int) else None,
        )
    return payload


def response_text(data: dict[str, Any]) -> str:
    """The assistant's reply text, with any ``` code fences removed."""
    choices = data.get("choices") or []
    if not choices:
        raise LLMError("empty response (no choices)")
    message = choices[0].get("message") or {}
    text = (message.get("content") or "").strip()
    if not text:
        raise LLMError(f"no text in response (finish_reason={choices[0].get('finish_reason')})")
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return fenced.group(1) if fenced else text


def call_with_retry(
    fn: Callable[[], T],
    max_retries: int | None = None,
    base_delay: float = BASE_BACKOFF_SECONDS,
    sleep: Callable[[float], None] | None = None,
) -> T:
    """Run fn, retrying retryable LLMErrors with exponential backoff (or the server's hint)."""
    # Looked up at call time (not as default args) so they can be patched.
    max_retries = MAX_RETRIES if max_retries is None else max_retries
    sleep = sleep or time.sleep
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except LLMError as exc:
            if not exc.retryable or attempt == max_retries:
                raise
            delay = exc.wait if exc.wait is not None else base_delay * 2**attempt
            if delay > MAX_WAIT_SECONDS:
                raise
            log.warning("%s; retrying in %.0fs", exc, delay)
            sleep(delay)
    raise AssertionError("unreachable")


def generate(body: dict[str, Any]) -> tuple[ExtractionResult, str]:
    """One call plus validation. Returns the result and the model that actually answered."""
    data = post_chat(body)
    text = response_text(data)
    try:
        result = ExtractionResult.model_validate_json(text)
    except ValidationError as exc:
        raise LLMError(f"output did not match the schema: {exc}") from exc
    return result, str(data.get("model", body["model"]))


def clean_actions(actions: list[ExtractedAction], email: Email) -> list[ExtractedAction]:
    """Normalize owners; drop (set to None) any owner who isn't a participant. Never guess."""
    participants = email.participants()
    cleaned: list[ExtractedAction] = []
    for action in actions:
        owner = normalize_address(action.owner) if action.owner else None
        if owner not in participants:
            owner = None
        cleaned.append(action.model_copy(update={"owner": owner}))
    return cleaned


def extract_actions_llm(email: Email) -> list[ExtractedAction]:
    """Send the email to OpenRouter and return the validated, cleaned actions."""
    body = build_request_body(email)
    result, used_model = call_with_retry(lambda: generate(body))
    if used_model != body["model"]:
        log.warning("Answered by fallback model %s (main model %s was unavailable)", used_model, body["model"])

    delay = config.llm_call_delay_seconds()
    if delay:
        time.sleep(delay)
    return clean_actions(result.actions, email)
