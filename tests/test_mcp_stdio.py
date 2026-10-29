from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from arlab.tasks.mcp_env import McpEnvironment, StdioTransport

SERVER = Path(__file__).resolve().parents[1] / "scripts" / "fake_mcp_server.py"


@pytest.fixture
def env():
    environment = McpEnvironment(StdioTransport([sys.executable, str(SERVER)]))
    try:
        yield environment
    finally:
        environment.close()


def test_stdio_transport_discovers_tools(env):
    assert env.tool_names() == ("knowledge_search", "service_metrics", "incident_changes")


def test_stdio_transport_calls_a_tool(env):
    payload = json.loads(env.call("service_metrics", {"service": "growth-feed"}))
    assert payload["service"] == "growth-feed"
    assert payload["error_rate"] == pytest.approx(0.42)
    assert payload["recent_error_rate"] > payload["baseline_error_rate"]


def test_stdio_transport_surfaces_tool_errors(env):
    with pytest.raises(RuntimeError, match="unknown tool"):
        env.call("does_not_exist", {})


def test_snapshot_is_empty_for_mcp(env):
    """MCP has no inspectable state, so these tasks must use the answer oracle."""
    assert env.snapshot() == {}


def test_sequential_calls_reuse_one_process(env):
    first = env.call("service_metrics", {"service": "a"})
    second = env.call("service_metrics", {"service": "b"})
    assert '"service": "a"' in first
    assert '"service": "b"' in second
