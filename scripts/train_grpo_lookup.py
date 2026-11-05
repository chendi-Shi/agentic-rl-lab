"""Self-contained GRPO on the lookup task set.

This is the fallback path, and it exists because verl 0.9.1 cannot be installed
next to vllm 0.8.5: verl needs huggingface-hub >= 1.0, which needs transformers 5.x,
which needs vllm >= 0.13, which needs torch 2.9 -- and this image's driver caps
CUDA at 12.8 with wheels that were unreachable at install time.

The pieces are all in arlab: rollouts come from a vLLM server through the same
runner used for the baseline, rewards are the same r_task, and the update is
arlab.train.grpo (group-normalised advantages, clipped surrogate, k3 KL).

Run:  python scripts/train_grpo_lookup.py --steps 12 --group-size 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arlab.policy.openai import ChatClient, LLMPolicy, system_prompt
from arlab.rollout.concurrent import collect_records
from arlab.rollout.runner import RunnerConfig, run_task
from arlab.store import TraceStore
from arlab.tasks.lookup_tasks import LookupEnv
from arlab.tasks.stage_io import load_tasks
from arlab.train.grpo import GRPOConfig, GRPOTrainer, group_advantages, token_logprobs
from arlab.train.rollout_batch import Episode, build_train_batch, messages_from_record


def as_dict(record: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(record):
        return dataclasses.asdict(record)
    return dict(record)


def collect_group(
    task,
    env_cls,
    policy_factory,
    cfg: RunnerConfig,
    *,
    store: TraceStore,
    group_size: int,
    workers: int = 8,
) -> tuple[list[Episode], list[float]]:
    episodes: list[Episode] = []
    rewards: list[float] = []
    system = system_prompt(task.tools)

    # Rollouts in a group are independent, so issuing them together is what the
    # box's vLLM server actually wants: 8 rollouts of one task took 40.8s in a
    # for-loop and 6.1s issued together, same rewards (scripts/bench_rollout.py).
    # Each rollout still gets its own sampling seed, because a shared seed made
    # all eight samples byte-identical even at temperature 0.9.
    records = collect_records(
        task,
        env_cls,
        policy_factory,
        cfg,
        group_size=group_size,
        workers=workers,
    )
    for record in records:
        record = as_dict(record)
        record["instruction"] = task.instruction
        episodes.append(
            Episode(
                task_id=task.task_id,
                messages=messages_from_record(record, system),
                reward=float(record["reward"]["r_task"]),
            )
        )
        rewards.append(float(record["reward"]["r_task"]))
    return episodes, rewards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("tasks/stage3-kept.json"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--base-url", default="http://127.0.0.1:6006")
    parser.add_argument("--served-model", default="qwen3-4b")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8, help="concurrent rollouts; 1 = sequential")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.001)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--store", type=Path, default=Path("traces/grpo-train.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("reports/grpo-lookup.json"))
    parser.add_argument(
        "--tag",
        default="",
        help="checkpoint name; required when several runs share one box",
    )
    args = parser.parse_args()

    loaded = load_tasks(args.tasks)
    if not loaded:
        raise SystemExit(f"no tasks in {args.tasks}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}
    )

    # LoRA keeps the trainable set small and makes the reference policy free:
    # the same weights with the adapter switched off are the reference.
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    trainer = GRPOTrainer(model, optimizer, GRPOConfig(kl_coef=args.kl_coef))
    store = TraceStore(args.store)

    def policy_factory(seed: int):
        """One client per rollout: the sampling seed is part of the request body."""

        client = ChatClient(
            base_url=args.base_url,
            model=args.served_model,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "seed": seed},
        )
        llm = LLMPolicy(client, max_tokens=512, temperature=args.temperature)
        return lambda task, steps: llm(task, steps)

    history: list[dict[str, float]] = []
    started = time.time()

    for step in range(args.steps):
        task = loaded[step % len(loaded)].task
        env_cls = LookupEnv
        cfg = RunnerConfig(config_id="grpo", seed=step, oracle="answer")

        episodes, rewards = collect_group(
            task, env_cls, policy_factory, cfg, store=store,
            group_size=args.group_size, workers=args.workers,
        )
        advantages = group_advantages(rewards, [task.task_id] * len(rewards))

        input_ids, attention, mask = None, None, None
        probe = build_train_batch(
            episodes,
            tokenizer,
            per_sample_advantages=advantages,
            old_logprobs=torch.zeros(len(episodes), 1),
            max_length=args.max_length,
            device=model.device,
        )
        input_ids, attention = probe.input_ids, probe.attention_mask

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention).logits
            old_logp = token_logprobs(logits, input_ids)
            with model.disable_adapter():
                ref_logits = model(input_ids=input_ids, attention_mask=attention).logits
                ref_logp = token_logprobs(ref_logits, input_ids)

        batch = build_train_batch(
            episodes,
            tokenizer,
            per_sample_advantages=advantages,
            old_logprobs=old_logp,
            ref_logprobs=ref_logp,
            max_length=args.max_length,
            device=model.device,
        )
        stats = trainer.step(batch)

        row = {
            "step": step,
            "task_id": task.task_id,
            "mean_reward": sum(rewards) / len(rewards),
            "reward_std": float(torch.tensor(rewards).std(unbiased=False)),
            "solved": sum(1 for r in rewards if r >= 0.999),
            **stats.as_dict(),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    ckpt = Path("checkpoints") / f"grpo-{args.tag or str(args.steps) + 'steps'}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    print(f"saved adapter -> {ckpt}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "steps": history,
                "config": vars(args) | {"tasks": str(args.tasks), "out": str(args.out)},
                "elapsed_s": round(time.time() - started, 1),
            },
            ensure_ascii=False,
            default=str,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
