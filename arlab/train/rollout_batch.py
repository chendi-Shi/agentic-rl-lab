"""Turn completed agent trajectories into a token batch for GRPO.

The trainer in :mod:`arlab.train.grpo` is pure Python by design (it takes a
duck-typed model), so everything that needs a tokenizer or a live server lives
here instead.

Two things learned on the box:

* The policy that rolls out is served by vLLM; the policy that trains is the same
  weights loaded with transformers. They differ slightly, which is why
  ``old_logprobs`` are recomputed from the training model instead of assumed.
* Every assistant message in the episode is trainable, not just the final answer:
  the tool calls are the behaviour being taught. ``apply_chat_template(...,
  return_assistant_tokens_mask=True)`` gives that mask for free; a template
  without support falls back to a coarse all-ones mask, which is worse but never
  silently wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from arlab.train.grpo import TokenBatch, expand_advantages


@dataclass
class Episode:
    """One finished rollout, kept as messages so it can be re-tokenized."""

    task_id: str
    messages: list[dict[str, Any]]
    reward: float


def messages_from_record(record: dict[str, Any], system: str) -> list[dict[str, Any]]:
    """Rebuild the chat transcript of a stored trajectory."""

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": record.get("instruction", "")},
    ]
    for step in record.get("steps", []):
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"tool": step.get("tool"), "args": step.get("args") or {}}, ensure_ascii=False
                ),
            }
        )
        messages.append({"role": "tool", "content": str(step.get("observation", ""))})
    if record.get("final_answer") is not None:
        messages.append({"role": "assistant", "content": str(record["final_answer"])})
    return messages


def encode_episode(
    messages: Sequence[dict[str, Any]], tokenizer, max_length: int = 2048
) -> tuple[list[int], list[int]]:
    """Tokenize a transcript and mark which tokens the model produced."""

    try:
        encoded = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
            add_generation_prompt=False,
        )
        ids = list(encoded["input_ids"])
        mask = [int(x) for x in encoded["assistant_masks"]]
        if not any(mask):  # template rendered but flagged nothing
            raise ValueError("empty assistant mask")
    except Exception:
        text = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=False
        )
        ids = list(tokenizer(text, return_attention_mask=False)["input_ids"])
        mask = [1] * len(ids)

    return ids[:max_length], mask[:max_length]


def pad_episodes(
    episodes: Sequence[Episode],
    tokenizer,
    max_length: int = 2048,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad into [B, T] tensors of ids, attention and completion mask."""

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    encoded = [encode_episode(e.messages, tokenizer, max_length) for e in episodes]
    width = max(len(ids) for ids, _ in encoded)

    ids_rows, att_rows, mask_rows = [], [], []
    for ids, mask in encoded:
        pad = width - len(ids)
        ids_rows.append(ids + [pad_id] * pad)
        att_rows.append([1] * len(ids) + [0] * pad)
        mask_rows.append(mask + [0] * pad)

    return (
        torch.tensor(ids_rows, dtype=torch.long, device=device),
        torch.tensor(att_rows, dtype=torch.long, device=device),
        torch.tensor(mask_rows, dtype=torch.long, device=device),
    )


def build_train_batch(
    episodes: Sequence[Episode],
    tokenizer,
    *,
    per_sample_advantages: Sequence[float],
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None = None,
    max_length: int = 2048,
    device: str | torch.device = "cpu",
) -> TokenBatch:
    """Assemble the padded batch the trainer expects."""

    input_ids, attention, mask = pad_episodes(episodes, tokenizer, max_length, device)
    batch = TokenBatch(
        input_ids=input_ids,
        attention_mask=attention,
        completion_mask=mask,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
    )
    batch.advantages = expand_advantages(list(per_sample_advantages), batch).to(
        input_ids.device
    )
    return batch


def read_records(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
