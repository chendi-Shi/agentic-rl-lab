from __future__ import annotations

import json

import pytest

from arlab.tasks.env import SyntheticBusinessEnv
from arlab.tasks.math_env import ArithmeticExecutor
from arlab.tasks.mcp_env import McpEnvironment
from arlab.tasks.replay_env import RecordingEnvironment, ReplayEnvironment


class FakeTransport:
    """Stands in for a real MCP stdio server; no subprocess in tests."""

    def __init__(self, tools, responses):
        self.tools = tools
        self.responses = responses
        self.closed = False

    def request(self, method, params):
        if method == "tools/list":
            return {"tools": [{"name": name} for name in self.tools]}
        if method == "tools/call":
            key = (params["name"], json.dumps(params["arguments"], sort_keys=True))
            return self.responses[key]
        raise AssertionError(f"unexpected method {method}")

    def close(self):
        self.closed = True


def test_mcp_environment_lists_and_calls_tools():
    transport = FakeTransport(
        tools=["knowledge_search", "service_metrics"],
        responses={
            ("service_metrics", json.dumps({"service": "payments"}, sort_keys=True)): {
                "content": [{"type": "text", "text": "error_rate=0.42"}]
            }
        },
    )
    env = McpEnvironment(transport)

    assert env.tool_names() == ("knowledge_search", "service_metrics")
    assert env.call("service_metrics", {"service": "payments"}) == "error_rate=0.42"
    assert env.snapshot() == {}
    env.close()
    assert transport.closed is True


def test_mcp_error_response_raises():
    transport = FakeTransport(
        tools=["t"],
        responses={("t", "{}"): {"isError": True, "content": [{"type": "text", "text": "boom"}]}},
    )
    env = McpEnvironment(transport)
    with pytest.raises(RuntimeError, match="boom"):
        env.call("t", {})


def test_recording_environment_roundtrip(tmp_path):
    path = tmp_path / "rec.json"
    recorder = RecordingEnvironment(inner=SyntheticBusinessEnv())
    recorder.call("orders.get", {"order_id": "A123"})
    recorder.dump(path)

    replayed = ReplayEnvironment.from_file(path)
    assert replayed.tool_names() == ("orders.get",)
    assert replayed.call("orders.get", {"order_id": "A123"}).startswith("order=A123")


def test_replay_miss_raises_by_default():
    env = ReplayEnvironment({})
    with pytest.raises(KeyError):
        env.call("orders.get", {"order_id": "A123"})
    assert env.misses == 1


def test_replay_can_fall_through_to_live_env():
    inner = SyntheticBusinessEnv()
    env = ReplayEnvironment({}, inner=inner, on_miss="live")
    assert env.call("orders.get", {"order_id": "A123"}).startswith("order=A123")


def test_replay_live_requires_inner():
    with pytest.raises(ValueError):
        ReplayEnvironment({}, on_miss="live")


def test_arithmetic_executor_accepts_plain_arithmetic():
    ex = ArithmeticExecutor()
    assert ex.run("17*23+9") == "400"
    assert ex.run("sum(range(1,51))") == "1275"
    assert ex.run("print(2**10-24)") == "1000"


def test_arithmetic_executor_rejects_escapes():
    ex = ArithmeticExecutor()
    for bad in ("__import__('os')", "open('x')", "(1).__class__", "[1][0]", "lambda: 1"):
        with pytest.raises(ValueError):
            ex.run(bad)


def test_arithmetic_executor_rejects_empty():
    with pytest.raises(ValueError):
        ArithmeticExecutor().run("   ")
