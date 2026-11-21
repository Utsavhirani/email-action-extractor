# Build prompt: Email Action Extractor + Evaluator

Paste everything below the line into Claude Code, from inside an empty project folder.

---

I'm learning to build AI applications. Help me build a small Python project: an LLM-based **email -> business action extractor**, and an **evaluator** that measures it. The evaluator is the main point of the project; keep the extractor deliberately simple. It is a library tested with pytest, with no web API.

Build it **in the order below**. After each step, run the tests, show me what you made and briefly explain the key idea, then **stop and wait for me to say "next"** before starting the next step.

## Constraints

- Python 3.11+, Pydantic v2, pytest, and **plain HTTP requests (`httpx`, no SDK)** to **free OpenRouter models** (`:free` IDs, OpenAI-style `chat/completions` endpoint).
- Use structured output (`response_format` with the Pydantic `ExtractionResult` JSON schema, `provider.require_parameters: true`) and always validate the reply with Pydantic. Pass fallback models in the `models` array so OpenRouter retries busy models within one request.
- Model ID comes from `LLM_MODEL` (fallbacks from `LLM_FALLBACK_MODELS`), with defaults set to current free models that support structured output (check `GET https://openrouter.ai/api/v1/models`).
- The API key comes from the environment (`OPENROUTER_API_KEY`). Never hard-code it. The free tier allows ~50 requests/day, so keep retries minimal.
- **LLM on/off switch**: env var `USE_LLM` (`1` = call the LLM, anything else / unset = off). Read it in one place (`core/config.py`). Off is the default, so nothing calls the API unless I turn it on.
- If a model's schema support rejects `date | None`, have the LLM output the due date as an ISO string (or null) and parse it into `ExtractedAction` afterwards.
- Free tier has tight rate limits: retry with exponential backoff on rate-limit errors (HTTP 429 / `RESOURCE_EXHAUSTED`), up to a fixed maximum number of retries (e.g. 5), then raise. Allow an optional delay between calls (`LLM_CALL_DELAY_SECONDS`, default 0) during eval runs.
- Email addresses are plain (`priya@example.com`, never `Priya <priya@example.com>`). Whenever addresses are compared (owner check, owner matching, grading), strip whitespace and lowercase them first.
- No extra frameworks (no LangChain, no FastAPI).
- Keep the code small and readable. Type hints everywhere, and short docstrings.
- I'm on **Windows / PowerShell**. Give every shell command in PowerShell syntax (e.g. `$env:USE_LLM="1"`), not bash.
- Before step 1, tell me how to set up the environment: create a venv, install with `pip install -e ".[dev]"` (pytest as an optional `dev` dependency), and set `OPENROUTER_API_KEY` in `.env`.

## Layout

```
core/
  __init__.py
  config.py      # reads USE_LLM, OPENROUTER_API_KEY, LLM_MODEL, LLM_FALLBACK_MODELS, LLM_TIMEOUT_SECONDS, LLM_CALL_DELAY_SECONDS
  models.py      # Email, ActionType, ExtractedAction, ExtractionResult
  prompts.py     # prompt text + PROMPT_VERSION string
  llm_extractor.py   # OpenRouter extraction over plain HTTP
  stub_extractor.py  # offline keyword-based extraction (no API)
  extractor.py   # extract_actions(email): picks llm or stub based on USE_LLM
evaluator/
  __init__.py
  dataset.py     # GoldenCase + load_dataset(path)
  matching.py    # pair predicted vs expected actions
  metrics.py     # precision / recall / F1 / field accuracy
  runner.py      # evaluate(dataset, extract_fn) -> Report
  data/golden.json
tests/
  test_models.py
  test_matching.py
  test_metrics.py
  test_runner.py
  test_eval_live.py   # marked @pytest.mark.eval
main.py
pyproject.toml        # deps + pytest config (see below)
.gitignore            # runs/, .env, .venv/, __pycache__/
```

pytest config in `pyproject.toml`: register the `eval` marker, set `addopts = "-m 'not eval'"` (a later `-m eval` on the command line overrides it), and set `pythonpath = ["."]` so tests can import `core` and `evaluator`.

## Step 1: Models (`core/models.py`)

- `Email`: `sender: str`, `receivers: list[str]`, `subject: str`, `date: datetime` (timezone-aware, enforced with `pydantic.AwareDatetime`), `body: str`, `files: list[str] = []`, plus a computed property `has_attachments` (`len(files) > 0`).
- `ActionType` (str Enum): `REPLY, SHARE_DOCUMENT, REVIEW_DOCUMENT, APPROVE, SCHEDULE_MEETING, PAY_INVOICE, COMPLETE_TASK, FOLLOW_UP, PROVIDE_INFORMATION, UNCLASSIFIED`.
- `ExtractedAction`: `action_type: ActionType`, `owner: str | None` (an email address), `due_date: date | None`, `action_description: str` (for people to read, **never graded**).
- `ExtractionResult`: `actions: list[ExtractedAction]` (the model's output schema).
- Tests: valid and invalid construction, and `has_attachments`.

## Step 2: Extractor (`core/config.py`, `core/prompts.py`, `core/*_extractor.py`, `core/extractor.py`)

- `extract_actions(email: Email) -> list[ExtractedAction]` is the single public entry point. It checks `USE_LLM` and sends the email to:
  - **on** → `llm_extractor` (OpenRouter)
  - **off** → `stub_extractor`: a tiny deterministic keyword matcher (e.g. "invoice" → `PAY_INVOICE`, "meeting"/"call" → `SCHEDULE_MEETING`, "review" + attachment → `REVIEW_DOCUMENT`; owner = first receiver; no due date). It's meant to be dumb: it lets the whole pipeline and the eval run with no API, and gives a baseline score the LLM should beat.
- Both extractors have the same signature, so the evaluator can't tell them apart.
- The rest of this step describes the LLM extractor:
- The prompt includes the sender, receivers, subject, date, file names and body, plus a one-line definition of each `ActionType`.
- Rules in the prompt:
  - list **every** action point; split compound requests into separate actions, one type each
  - don't duplicate the same request; don't infer actions that weren't asked for
  - return an empty list if the email asks for nothing
  - use `UNCLASSIFIED` when an action exists but no type fits
  - resolve relative dates ("by Friday") against `email.date` and output ISO dates
  - `owner` must be one of the email's participants
- Light checks after the call: if `owner` isn't one of sender/receivers, set it to `None`. Never guess.
- `main.py`: build one sample email, call `extract_actions`, print which mode is active (LLM or stub) and the actions.
- Tests: the stub extractor (deterministic, no API), and that the switch picks the right extractor (monkeypatch the env var; mock the LLM call, never hit the network in unit tests).

## Step 3: Golden dataset (`evaluator/dataset.py`, `evaluator/data/golden.json`)

- `GoldenCase`: `id: str`, `email: Email`, `expected: list[ExtractedAction]`, `notes: str = ""`.
- `load_dataset(path) -> list[GoldenCase]`, validated with Pydantic.
- Write **about 15 cases** covering: single action, several actions, a compound request that has to be split, no action (FYI or newsletter), a relative deadline, attachments, several receivers (owner ambiguity), and something that should be `UNCLASSIFIED`.
- `action_description` in the expected answers can be a short placeholder, since it isn't graded.
- Test: the file loads and every case validates.

## Step 4: Matching (`evaluator/matching.py`)

- `match_actions(predicted, expected) -> MatchResult` with `pairs: list[tuple[pred, exp]]`, `false_positives: list[pred]` and `false_negatives: list[exp]`.
- Rule, in two passes so list order doesn't matter:
  1. Pair each expected action with an unused predicted action that has the **same `action_type` and the same `owner`**.
  2. Pair the remaining expected actions with any unused predicted action of the **same `action_type`**.
- (A greedy single pass is order-dependent: expected `[A x, A y]` vs predicted `[A y, A z]` would pair x↔y and y↔z, giving owner accuracy 0/2 instead of 1/2.)
- A wrong type never matches (strict, no partial credit).
- Tests: perfect match, an empty prediction, an extra prediction, duplicate types, reordered lists, the owner tie-break, and the order-dependence example above.

## Step 5: Metrics (`evaluator/metrics.py`)

- Field grading on matched pairs: `owner` and `due_date` with exact match. **`action_description` is explicitly excluded**: keep a `GRADED_FIELDS` constant so this is visible.
- **Micro-averaged** precision, recall and F1 from TP/FP/FN totals summed over all cases. Handle zero denominators without crashing (define the result and document the choice, including field accuracy when there are no matched pairs).
- Test with the worked example: expected `[REVIEW_DOCUMENT owner=priya@example.com due=2026-11-27, SCHEDULE_MEETING owner=priya@example.com due=None]`, predicted `[REVIEW_DOCUMENT owner=priya@example.com due=2026-11-28, REPLY owner=priya@example.com due=None]` → TP=1, FP=1, FN=1, P=0.5, R=0.5, owner accuracy 1/1, due-date accuracy 0/1.

## Step 6: Runner + report (`evaluator/runner.py`)

- `evaluate(dataset, extract_fn: Callable[[Email], list[ExtractedAction]]) -> Report`. The evaluator must **not** import the extractor.
- For each case: run `extract_fn` (catch exceptions and record the case as errored), match, grade, and build a `CaseResult` with TP/FP/FN, field misses and readable failure reasons. Include the predicted `action_description` in failure details.
- `Report`: precision, recall, F1, owner accuracy, due-date accuracy, perfect-case rate, correct "no action" cases (shown as `x / total no-action cases`), error count, and the list of `CaseResult`.
- `Report.summary() -> str` for printing, and `Report.save(path)` to write JSON including a timestamp, the extractor mode / model name, and the prompt version. Both the mode label and `prompt_version` are passed in as arguments (the caller reads `PROMPT_VERSION` from `core/prompts.py`), since the evaluator doesn't import `core`.
- Tests with **fake extractors**: `perfect_fn` (looks up the expected answer for each email from the dataset, e.g. a dict keyed by subject; everything 1.0), `empty_fn` (recall 0, no crash), `extra_fn` (recall 1.0, precision < 1.0) and `crashing_fn` (error counted, run continues).

## Step 7: Live eval (`tests/test_eval_live.py`)

- Marked `@pytest.mark.eval`, so a plain `pytest` skips it and `pytest -m eval` runs it.
- Runs `extract_actions` over `golden.json`, prints `report.summary()`, and saves the report to `runs/`. The report records which mode was used (`stub` or the LLM model ID), so stub and LLM runs are never confused.
- With `USE_LLM` off, the eval runs against the stub (baseline score, no API). With it on, it runs against the OpenRouter model.
- Asserts placeholder thresholds (`recall >= 0.0`). I'll set real thresholds after looking at the first run.
- Configure pytest so `eval` tests are skipped by default.

## Done when

- `pytest` passes quickly with no API calls, whatever `USE_LLM` is set to.
- `pytest -m eval -s` with `USE_LLM` off prints the stub's baseline report.
- `$env:USE_LLM="1"; pytest -m eval -s` (PowerShell; bash: `USE_LLM=1 pytest -m eval -s`) runs the real LLM eval and prints a readable report.
- `python main.py` prints the active mode and the actions for a sample email.
