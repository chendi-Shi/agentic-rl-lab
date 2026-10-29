from __future__ import annotations

import json

import pytest

from arlab.store import TraceRecord, TraceStore, write_jsonl
from arlab.store.trace import rollout_key
from arlab.tasks.schema import Step, ToolCall, Trajectory


def make_record(task_id: str = "t1", total: float = 1.0) -> TraceRecord:
    traj = Trajectory(
        task_id=task_id,
        steps=[
            Step(tool_call=ToolCall("orders.get", {"order_id": "A123"}), observation="ok", tokens_in=11, tokens_out=3),
            Step(tool_call=ToolCall("orders.refund", {"order_id": "A123"}), observation="done", tokens_in=7),
        ],
        final_answer="ok",
    )
    return TraceRecord(
        key=rollout_key(task_id, "p", "c", 0),
        task_id=task_id,
        policy="p",
        config_id="c",
        seed=0,
        trajectory=traj,
        reward={"total": total},
        meta={"wall_seconds": 0.5},
    )


def test_roundtrip_preserves_steps_tokens_and_reward(tmp_path):
    store = TraceStore(tmp_path / "t.jsonl")
    record = make_record()
    store.append(record)

    loaded = store.load()
    assert len(loaded) == 1
    back = loaded[0]
    assert back.key == record.key
    assert back.reward == {"total": 1.0}
    assert len(back.trajectory) == 2
    assert back.trajectory.tokens() == (18, 3)
    assert back.trajectory.steps[0].tool_call.name == "orders.get"
    assert back.trajectory.steps[0].tool_call.args == {"order_id": "A123"}
    assert back.meta["wall_seconds"] == 0.5


def test_done_keys_and_grouping(tmp_path):
    store = TraceStore(tmp_path / "t.jsonl")
    store.append(make_record("t1", 1.0))
    store.append(make_record("t2", 0.0))

    assert store.done_keys() == {rollout_key("t1", "p", "c", 0), rollout_key("t2", "p", "c", 0)}
    assert store.rewards_by_task() == {"t1": [1.0], "t2": [0.0]}


def test_missing_file_loads_empty(tmp_path):
    store = TraceStore(tmp_path / "nope.jsonl")
    assert store.load() == []
    assert store.done_keys() == set()


def test_corrupt_line_reports_location(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"key": "k"}\n', encoding="utf-8")
    store = TraceStore(path)
    with pytest.raises(ValueError, match="corrupt trace"):
        store.load()


def test_write_jsonl(tmp_path):
    path = tmp_path / "out" / "rows.jsonl"
    write_jsonl(path, [{"a": 1}, {"b": 2}])
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": 2}]
