"""Command builders for the GPU day.

These are *templates*. The VRAM pre-flight can be trusted offline; the exact
verl override keys cannot -- they change between releases. ``VERL_OVERRIDES``
below isolates every key in one dict so the first GPU session only needs to fix
that dict, not the whole script.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from arlab.train.config import TrainConfig, estimate_vram_gb

#: verl trainer overrides. Verify each key against the installed version.
VERL_OVERRIDES: dict[str, str] = {
    "algorithm.adv_estimator": "grpo",
    "actor_rollout_ref.actor.optim.lr": "{lr}",
    "actor_rollout_ref.actor.ppo_mini_batch_size": "{mini_batch}",
    "actor_rollout_ref.actor.entropy_coeff": "{entropy_coef}",
    "actor_rollout_ref.model.lora_rank": "{lora_rank}",
    "actor_rollout_ref.model.lora_alpha": "{lora_alpha}",
    "actor_rollout_ref.rollout.n": "{group_size}",
    "actor_rollout_ref.rollout.gpu_memory_utilization": "{vllm_gpu_util}",
    "actor_rollout_ref.rollout.tensor_model_parallel_size": "{vllm_tp}",
    "actor_rollout_ref.rollout.max_num_seqs": "{rollout_concurrency}",
    "actor_rollout_ref.rollout.response_length": "{max_response_tokens}",
    "actor_rollout_ref.rollout.prompt_length": "{max_prompt_tokens}",
    "data.train_batch_size": "{train_batch}",
    "trainer.total_epochs": "{total_epochs}",
    "trainer.save_freq": "{save_freq}",
    "trainer.experiment_name": "{experiment}",
}


def build_verl_command(cfg: TrainConfig, *, module: str = "verl.trainer.main_ppo") -> list[str]:
    """Assemble the trainer invocation with all overrides rendered."""

    values = cfg.to_json()
    command = [
        "python",
        "-m",
        module,
        f"data.train_files={cfg.tasks_file}",
        f"actor_rollout_ref.model.path={cfg.model}",
    ]
    for key, template in VERL_OVERRIDES.items():
        command.append(f"{key}={template.format(**values)}")
    return command


def build_vllm_command(cfg: TrainConfig, *, port: int = 8000) -> list[str]:
    """Standalone vLLM server, for the evaluation pass that runs outside verl."""

    return [
        "vllm",
        "serve",
        cfg.model,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        f"{cfg.vllm_gpu_util}",
        "--max-model-len",
        str(cfg.max_prompt_tokens + cfg.max_response_tokens),
        "--max-num-seqs",
        str(cfg.rollout_concurrency),
        "--tensor-parallel-size",
        str(cfg.vllm_tp),
    ]


def render(command: list[str]) -> str:
    """One-line shell form, copy-pasteable on the rented box."""

    return " ".join(shlex.quote(part) for part in command)


@dataclass
class CostEstimate:
    hours: float
    hourly_price: float
    total: float


def estimate_cost(cfg: TrainConfig, hours: float, hourly_price: float) -> CostEstimate:
    return CostEstimate(hours=hours, hourly_price=hourly_price, total=round(hours * hourly_price, 2))


def dry_run(cfg: TrainConfig, *, gpu_gb: float = 80.0) -> dict[str, Any]:
    """Everything worth checking before paying for a card, in one dict."""

    return {
        "vram": estimate_vram_gb(cfg),
        "vram_budget_gb": gpu_gb,
        "verl_command": render(build_verl_command(cfg)),
        "vllm_command": render(build_vllm_command(cfg)),
        "problems": cfg.validate(),
    }
