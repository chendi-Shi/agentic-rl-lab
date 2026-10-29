from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from arlab.policy import INVALID_TOOL, ChatClient, LLMPolicy, build_messages, parse_action
from arlab.rollout import FinalAnswer
from arlab.tasks.schema import Step, Task, ToolCall


class FakeHandler(BaseHTTPRequestHandler):
    reply = '{"tool": "python", "args": {"code": "1+1"}}'
    captured: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
        length = int(self.headers.get("content-length", 0))
        FakeHandler.captured.append(json.loads(self.rfile.read(length)))
        payload = {
            "choices": [{"message": {"content": FakeHandler.reply}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # silence the test log
        return


@pytest.fixture
def server():
    FakeHandler.captured = []
    FakeHandler.reply = '{"tool": "python", "args": {"code": "1+1"}}'
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def sample_task() -> Task:
    return Task(
        task_id="t",
        instruction="compute 1+1",
        required_fields={"answer": "2"},
        max_steps=6,
        tools=("python",),
    )


# ------------------------------------------------------------------ parsing


def test_parse_fenced_json():
    action = parse_action('sure\n```json\n{"tool": "python", "args": {"code": "1"}}\n```')
    assert isinstance(action, ToolCall)
    assert action.name == "python"
    assert action.args == {"code": "1"}


def test_parse_json_embedded_in_prose():
    action = parse_action('I will call {"tool": "python", "args": {"code": "2"}} now.')
    assert isinstance(action, ToolCall)
    assert action.name == "python"


def test_parse_final_answer():
    action = parse_action('{"final_answer": "400"}')
    assert isinstance(action, FinalAnswer)
    assert action.text == "400"


def test_parse_accepts_name_and_arguments_aliases():
    action = parse_action('{"name": "python", "arguments": {"code": "3"}}')
    assert isinstance(action, ToolCall)
    assert action.name == "python"


@pytest.mark.parametrize("text", ["", "no json here", '{"tool": 1, "args": {}}', "[1,2]", "{broken"])
def test_parse_garbage_becomes_invalid_call(text):
    action = parse_action(text)
    assert isinstance(action, ToolCall)
    assert action.name == INVALID_TOOL


# ---------------------------------------------------------------- messages


def test_build_messages_masks_old_observations():
    task = sample_task()
    steps = [
        Step(tool_call=ToolCall("python", {"code": "1"}), observation="A" * 500),
        Step(tool_call=ToolCall("python", {"code": "2"}), observation="B" * 500),
        Step(tool_call=ToolCall("python", {"code": "3"}), observation="C" * 500),
        Step(tool_call=ToolCall("python", {"code": "4"}), observation="D" * 500),
        Step(tool_call=ToolCall("python", {"code": "5"}), observation="E" * 500),
    ]

    messages = build_messages(task, steps, keep_recent=2, max_observation_chars=100)

    assert messages[0]["role"] == "system"
    assert "python" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": task.instruction}
    observations = [m["content"] for m in messages if m["content"].startswith("observation:")]
    assert len(observations) == 5
    assert "masked" in observations[0]  # old ones are compressed
    assert len(observations[-1]) <= len("observation: ") + 100 + 4  # recent one is truncated, not dropped


# ------------------------------------------------------------------ client


def test_client_posts_and_reads_usage(server):
    client = ChatClient(base_url=server, model="qwen3-8b", timeout=10)
    text, usage = client.complete([{"role": "user", "content": "hi"}], max_tokens=8)

    assert "python" in text
    assert usage.prompt_tokens == 11
    assert usage.completion_tokens == 7
    sent = FakeHandler.captured[0]
    assert sent["model"] == "qwen3-8b"
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


def test_client_reports_unreachable_endpoint():
    client = ChatClient(base_url="http://127.0.0.1:1", timeout=2)
    with pytest.raises(RuntimeError, match="cannot reach"):
        client.complete([{"role": "user", "content": "hi"}])


def test_llm_policy_returns_parsed_action(server):
    policy = LLMPolicy(ChatClient(base_url=server, timeout=10), max_tokens=16)
    action = policy(sample_task(), [])

    assert isinstance(action, ToolCall)
    assert action.name == "python"
    assert policy.last_usage.prompt_tokens == 11


def test_llm_policy_survives_bad_reply(server):
    FakeHandler.reply = "I refuse."
    policy = LLMPolicy(ChatClient(base_url=server, timeout=10))
    action = policy(sample_task(), [])

    assert isinstance(action, ToolCall)
    assert action.name == INVALID_TOOL
