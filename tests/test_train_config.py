from __future__ import annotations

import json

import pytest

from arlab.train import TrainConfig, build_verl_command, build_vllm_command, dry_run, estimate_vram_gb, preflight, render


def test_config_roundtrip(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"model": "Qwen/Qwen3-4B", "params_b": 4.0, "group_size": 8}), encoding="utf-8")

    cfg = TrainConfig.from_json(path)

    assert cfg.model == "Qwen/Qwen3-4B"
    assert cfg.group_size == 8
    assert cfg.train_batch == 32  # default preserved


def test_validate_catches_bad_batching():
    cfg = TrainConfig(group_size=8, train_batch=30)
    assert any("divisible" in p for p in cfg.validate())


def test_validate_catches_tiny_group():
    assert any("group_size" in p for p in TrainConfig(group_size=1).validate())


def test_estimate_grows_with_model_size():
    small = estimate_vram_gb(TrainConfig(params_b=4.0))["total"]
    large = estimate_vram_gb(TrainConfig(params_b=8.0))["total"]
    assert large > small


def test_preflight_rejects_config_that_does_not_fit():
    cfg = TrainConfig(params_b=32.0, rollout_concurrency=8)
    warnings = preflight(cfg, gpu_gb=80.0)
    assert any("estimated" in w for w in warnings)


def test_preflight_accepts_4b_on_80gb():
    assert preflight(TrainConfig(params_b=4.0), gpu_gb=80.0) == []


def test_preflight_accepts_8b_on_80gb():
    """Regression: an inflated KV coefficient once reported 8B as 81GB and refused.

    8B bf16 weights + 4-bit reference + a separate vLLM copy + KV for 8x3584
    tokens is ~53GB, which fits an 80GB card comfortably.
    """
    cfg = TrainConfig(params_b=8.0, rollout_concurrency=8)
    estimate = estimate_vram_gb(cfg)

    assert 45.0 < estimate["total"] < 60.0
    assert preflight(cfg, gpu_gb=80.0) == []


def test_preflight_accepts_14b_with_quantised_rollout():
    cfg = TrainConfig(params_b=14.0, rollout_4bit=True, rollout_concurrency=4)
    assert preflight(cfg, gpu_gb=80.0) == []


def test_preflight_rejects_32b_even_at_4bit():
    cfg = TrainConfig(params_b=32.0, train_4bit=True, rollout_4bit=True, rollout_concurrency=8)
    warnings = preflight(cfg, gpu_gb=80.0)
    assert any("estimated" in w for w in warnings)


def test_kv_cache_scales_with_concurrency_not_squared():
    low = estimate_vram_gb(TrainConfig(params_b=8.0, rollout_concurrency=2))["kv_cache"]
    high = estimate_vram_gb(TrainConfig(params_b=8.0, rollout_concurrency=8))["kv_cache"]
    assert high == pytest.approx(low * 4, rel=0.02)  # values are rounded to 2dp


def test_verl_command_carries_overrides():
    command = build_verl_command(TrainConfig(params_b=4.0, lora_rank=32))
    joined = " ".join(command)
    assert "verl.trainer.main_ppo" in joined
    assert "algorithm.adv_estimator=grpo" in joined
    assert "lora_rank=32" in joined
    assert "rollout.n=8" in joined


def test_vllm_command_shape():
    cfg = TrainConfig()
    command = build_vllm_command(cfg)
    assert command[:3] == ["vllm", "serve", cfg.model]
    assert "--gpu-memory-utilization" in command


def test_render_quotes_shell_specials():
    assert render(["a b", "c"]) == "'a b' c"


def test_dry_run_bundles_everything():
    report = dry_run(TrainConfig(), gpu_gb=80.0)
    assert set(report) == {"vram", "vram_budget_gb", "verl_command", "vllm_command", "problems"}
    assert report["vram"]["total"] > 0
