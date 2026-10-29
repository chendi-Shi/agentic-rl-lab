"""Training configuration plus a VRAM pre-flight estimator.

The estimator is deliberately crude and explicit: its job is to stop you from
renting a card that cannot hold the configuration, not to predict throughput.
Every assumption is a named number in :func:`estimate_vram_gb`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    # model
    model: str = "Qwen/Qwen3-4B"
    params_b: float = 4.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # optimisation
    lr: float = 1e-6
    group_size: int = 8
    train_batch: int = 32
    mini_batch: int = 8
    kl_coef: float = 0.001
    entropy_coef: float = 0.0
    total_epochs: int = 1
    save_freq: int = 50
    # rollout
    max_steps: int = 8
    max_prompt_tokens: int = 2048
    max_response_tokens: int = 512
    rollout_concurrency: int = 8
    vllm_gpu_util: float = 0.55
    vllm_tp: int = 1
    ref_model_4bit: bool = True
    # bookkeeping
    experiment: str = "arlab-grpo"
    tasks_file: str = "tasks/stage1.json"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.group_size < 2:
            problems.append("group_size < 2: GRPO needs at least two samples per task")
        if self.train_batch % self.group_size != 0:
            problems.append(
                f"train_batch ({self.train_batch}) must be divisible by group_size ({self.group_size})"
            )
        if self.mini_batch > self.train_batch:
            problems.append("mini_batch > train_batch")
        if not 0.1 <= self.vllm_gpu_util <= 0.9:
            problems.append("vllm_gpu_util outside [0.1, 0.9]")
        if self.max_steps <= 0:
            problems.append("max_steps must be positive")
        return problems


def estimate_vram_gb(cfg: TrainConfig) -> dict[str, float]:
    """Rough peak-VRAM budget in GB.

    Assumptions (all ballpark, documented so they can be corrected on real runs):
      * bf16 weights = 2 bytes/param
      * LoRA trainable + optimizer states ~ 0.55 GB per billion params
      * reference model in bf16, or half that when quantised to 4-bit
      * vLLM shares a *separate* copy of the weights, plus KV cache
      * KV cache ~ 0.14 GB per 1k tokens per billion params (GQA 8 heads, fp16)
    """

    weights = cfg.params_b * 2.0
    lora_state = cfg.params_b * 0.55
    reference = weights * (0.5 if cfg.ref_model_4bit else 1.0)
    rollout_weights = weights
    kv = (cfg.params_b * 0.14) * (
        (cfg.max_prompt_tokens + cfg.max_response_tokens) * cfg.rollout_concurrency / 1000.0
    )
    activations = 2.0 + 0.3 * cfg.params_b
    return {
        "weights": round(weights, 2),
        "lora_and_optimizer": round(lora_state, 2),
        "reference": round(reference, 2),
        "rollout_weights": round(rollout_weights, 2),
        "kv_cache": round(kv, 2),
        "activations_misc": round(activations, 2),
        "total": round(
            weights + lora_state + reference + rollout_weights + kv + activations, 2
        ),
    }


def preflight(cfg: TrainConfig, gpu_gb: float = 80.0) -> list[str]:
    """Return warnings; an empty list means the configuration fits."""

    warnings = list(cfg.validate())
    estimate = estimate_vram_gb(cfg)
    if estimate["total"] > gpu_gb * 0.92:
        warnings.append(
            f"estimated {estimate['total']:.0f}GB > usable {gpu_gb * 0.92:.0f}GB on a {gpu_gb:.0f}GB card: "
            "reduce params_b, rollout_concurrency, or set ref_model_4bit"
        )
    return warnings
