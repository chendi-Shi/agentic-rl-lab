"""OpenAI-compatible chat client and the policy that drives it.

Uses only the standard library so the file can be copied onto a rented box
without installing anything. Verified against a fake server in tests; the real
endpoint is vLLM's ``/v1/chat/completions``.

Protocol the model must speak (enforced by ``parse_action``):

    {"tool": "python", "args": {"code": "17*23+9"}}
    {"final_answer": "400"}

Anything else becomes an ``__invalid__`` tool call, which the environment turns
into an ``ERROR:`` observation. That keeps a malformed step *in* the trajectory
as a penalty instead of silently ending the episode.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from arlab.rollout.runner import FinalAnswer
from arlab.tasks.schema import Step, Task, ToolCall

INVALID_TOOL = "__invalid__"

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
# Note: the JSON braces below are literal. Do not run .format() over this string.
_SYSTEM = (
    "You are a tool-using agent. Reply with exactly one JSON object and nothing else.\n"
    'To call a tool: {"tool": "<name>", "args": {...}}\n'
    'To finish: {"final_answer": "<answer>"}\n'
    "Available tools: <TOOLS>"
)


def system_prompt(tools: Sequence[str]) -> str:
    """Render the system prompt; tool names are substituted, braces are literal."""

    return _SYSTEM.replace("<TOOLS>", ", ".join(tools) or "(none)")


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatClient:
    """Minimal blocking client for an OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        model: str = "qwen3-8b",
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        max_retries: int = 2,
        backoff: float = 0.3,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.backoff = backoff
        # Server-side switches that are not part of the OpenAI schema, e.g.
        # {"chat_template_kwargs": {"enable_thinking": False}} for Qwen3.
        self.extra_body = dict(extra_body or {})

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> tuple[str, Usage]:
        """POST once, retrying transient failures.

        A single dropped connection must not kill a 40-minute sweep on a billed
        box, so 5xx and transport errors are retried with exponential backoff.
        4xx is not: a bad request will fail the same way every time.
        """

        payload = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        payload.update(self.extra_body)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                body = self._post(payload)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:200]
                error = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code < 500:
                    raise error from exc
                last_error = error
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"cannot reach {self.base_url}: {exc.reason}")
            except TimeoutError as exc:
                last_error = RuntimeError(f"timeout after {self.timeout}s: {exc}")

            if attempt < self.max_retries:
                time.sleep(self.backoff * (2**attempt))
        else:
            raise last_error or RuntimeError("request failed")

        choice = (body.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = body.get("usage") or {}
        return text, Usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )


_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove a reasoning model's ``<think>...</think>`` blocks.

    Qwen3 and friends emit their reasoning *before* the action, and that text can
    contain braces (code, JSON examples), which makes naive brace matching pick
    the wrong object. An unterminated block means the model never finished
    thinking, so there is no action to find.
    """

    cleaned = _THINK.sub("", text or "")
    return "" if _THINK_OPEN.search(cleaned) else cleaned


def _repair_missing_quote(candidate: str) -> dict[str, Any] | None:
    """Re-insert the closing quote a model forgot before the final brace(s).

    Observed with Qwen3-4B: ``{"tool": "python", "args": {"code": "x = 1}}`` --
    the code string is never closed, so the braces land inside it. The intent is
    unambiguous, so try the last few brace positions and accept the first repair
    that parses *and* looks like an action.
    """

    positions = [i for i, ch in enumerate(candidate) if ch == "}"]
    for index in positions[-3:]:
        attempt = candidate[:index] + '"' + candidate[index:]
        try:
            payload = json.loads(attempt, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and ({"tool", "name", "final_answer"} & set(payload)):
            return payload
    return None


def parse_action(text: str) -> ToolCall | FinalAnswer:
    """Turn one model reply into an action. Never raises."""

    raw = (text or "")[:200]
    candidate = strip_think(text or "")
    match = _FENCE.search(candidate)
    if match:
        candidate = match.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return ToolCall(INVALID_TOOL, {"raw": raw})
        candidate = candidate[start : end + 1]

    try:
        # strict=False: LLMs habitually put real newlines inside JSON strings
        # (code snippets), which is invalid JSON but unambiguous.
        payload = json.loads(candidate, strict=False)
    except json.JSONDecodeError:
        payload = _repair_missing_quote(candidate)

    if not isinstance(payload, dict):
        return ToolCall(INVALID_TOOL, {"raw": raw})
    if "final_answer" in payload:
        return FinalAnswer(str(payload["final_answer"]))
    tool = payload.get("tool") or payload.get("name")
    args = payload.get("args") or payload.get("arguments") or {}
    if not isinstance(tool, str) or not isinstance(args, dict):
        return ToolCall(INVALID_TOOL, {"raw": raw})
    return ToolCall(tool, args)


def build_messages(
    task: Task,
    steps: Sequence[Step],
    *,
    max_observation_chars: int = 600,
    keep_recent: int = 3,
) -> list[dict[str, str]]:
    """Render the trajectory for the model.

    Older observations are masked rather than dropped: the *fact* that a tool was
    called and what it returned (roughly) still matters, but the bytes do not.
    This is the cheapest form of context budgeting and it is applied before any
    token limit is hit, not after.
    """

    messages = [
        {"role": "system", "content": system_prompt(task.tools)},
        {"role": "user", "content": task.instruction},
    ]
    cutoff = len(steps) - keep_recent
    for index, step in enumerate(steps):
        if step.tool_call is None:
            continue
        call = step.tool_call
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps({"tool": call.name, "args": dict(call.args)}, ensure_ascii=False),
            }
        )
        observation = step.observation or ""
        if index < cutoff and len(observation) > 80:
            observation = observation[:80] + f" ...[{len(observation)} chars masked]"
        elif len(observation) > max_observation_chars:
            observation = observation[:max_observation_chars] + " ..."
        messages.append({"role": "user", "content": f"observation: {observation}"})
    return messages


class LLMPolicy:
    """``Policy`` implementation backed by a chat endpoint."""

    def __init__(
        self,
        client: ChatClient,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        max_observation_chars: int = 600,
        keep_recent: int = 3,
        error_mode: str = "invalid",
    ) -> None:
        if error_mode not in {"invalid", "raise"}:
            raise ValueError(f"unknown error_mode: {error_mode!r}")
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_observation_chars = max_observation_chars
        self.keep_recent = keep_recent
        self.error_mode = error_mode
        self.last_usage: Usage = Usage()

    def __call__(self, task: Task, steps: list[Step]) -> ToolCall | FinalAnswer:
        messages = build_messages(
            task,
            steps,
            max_observation_chars=self.max_observation_chars,
            keep_recent=self.keep_recent,
        )
        try:
            text, usage = self.client.complete(
                messages, max_tokens=self.max_tokens, temperature=self.temperature
            )
        except RuntimeError as exc:
            # One bad request should cost one step, not the whole sweep. The
            # invalid call is recorded and scored, so the loss shows up as a
            # lower reward rather than a crashed run.
            if self.error_mode == "raise":
                raise
            self.last_usage = Usage()
            return ToolCall(INVALID_TOOL, {"error": str(exc)[:200]})
        self.last_usage = usage
        return parse_action(text)


def annotate_tokens(steps: list[Step], usage: Usage) -> None:
    """Attribute token usage to the most recent step (called by the driver)."""

    if not steps:
        return
    steps[-1].tokens_in += usage.prompt_tokens
    steps[-1].tokens_out += usage.completion_tokens
