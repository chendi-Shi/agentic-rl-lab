"""A self-contained GRPO step: advantages, clipped policy loss, KL penalty.

Why not just use verl? Because verl's *multi-turn agent* API is the part that
changes between releases, and it is the only part we cannot test offline. The
GRPO math underneath it is small, stable, and completely testable on CPU -- so
it lives here, and the framework becomes optional instead of load-bearing.

Two deliberate choices:

* **Duck-typed model.** The trainer needs ``model(input_ids=..., attention_mask=...)
  -> logits`` and ``model.parameters()``. That is all. Tests drive it with a toy
  module; the card drives it with a PEFT-wrapped HF model. No transformers import
  here, so an upgrade cannot break the training step.
* **LoRA means the reference model is free.** With adapters, the reference policy
  is the same weights with the adapter disabled, so ``ref_logprobs`` can be
  computed by the caller with no second copy of the model. On a 24GB card that
  is the difference between fitting and not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class GRPOConfig:
    clip: float = 0.2
    kl_coef: float = 0.001
    lr: float = 1e-5
    epochs: int = 1
    max_grad_norm: float = 1.0
    eps: float = 1e-6


@dataclass
class GRPOStats:
    loss: float
    policy_loss: float
    kl: float
    mean_ratio: float
    clip_fraction: float
    mean_advantage: float
    n_tokens: int

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "policy_loss": self.policy_loss,
            "kl": self.kl,
            "mean_ratio": self.mean_ratio,
            "clip_fraction": self.clip_fraction,
            "mean_advantage": self.mean_advantage,
            "n_tokens": float(self.n_tokens),
        }


def group_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[str],
    eps: float = 1e-6,
) -> list[float]:
    """Normalise rewards within each group of samples for the same task.

    A group whose rewards are all equal collapses to zero advantage -- that is
    the failure mode of an all-solved or all-failed task set, and it is reported
    rather than hidden, because a silently zero gradient looks like a slow run.
    """

    if len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must be the same length")
    if not rewards:
        raise ValueError("empty batch")

    buckets: dict[str, list[int]] = {}
    for index, group in enumerate(group_ids):
        buckets.setdefault(group, []).append(index)

    advantages = [0.0] * len(rewards)
    for indices in buckets.values():
        values = [float(rewards[i]) for i in indices]
        mean = sum(values) / len(values)
        if len(values) < 2:
            continue
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        if std <= eps:
            continue
        for i in indices:
            advantages[i] = (float(rewards[i]) - mean) / (std + eps)
    return advantages


def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token log-probability of the tokens that actually appeared.

    ``logits`` is [B, T, V] and ``input_ids`` is [B, T]; the result is [B, T-1],
    aligned with ``input_ids[:, 1:]``.
    """

    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, T, V], got {tuple(logits.shape)}")
    if input_ids.shape[:2] != logits.shape[:2]:
        raise ValueError("input_ids and logits disagree on batch or sequence length")

    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    target = input_ids[:, 1:].unsqueeze(-1)
    return logp.gather(-1, target).squeeze(-1)


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    cfg: GRPOConfig,
    ref_logp: torch.Tensor | None = None,
) -> tuple[torch.Tensor, GRPOStats]:
    """Clipped surrogate objective with a k3 KL penalty, averaged over tokens."""

    if logp.shape != old_logp.shape:
        raise ValueError(f"logp {tuple(logp.shape)} vs old_logp {tuple(old_logp.shape)}")
    if mask.shape != logp.shape:
        raise ValueError(f"mask {tuple(mask.shape)} vs logp {tuple(logp.shape)}")

    mask = mask.float()
    n_tokens = int(mask.sum().item())
    if n_tokens == 0:
        raise ValueError("no completion tokens to train on")

    log_ratio = logp - old_logp
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)

    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    policy_loss = -(surrogate * mask).sum() / n_tokens

    if ref_logp is not None:
        # k3 estimator: exp(d) - d - 1 with d = ref - policy; always >= 0.
        delta = ref_logp - logp
        kl = torch.exp(delta) - delta - 1.0
        kl_loss = (kl * mask).sum() / n_tokens
    else:
        kl = torch.zeros_like(logp)
        kl_loss = torch.zeros((), dtype=logp.dtype)

    total = policy_loss + cfg.kl_coef * kl_loss

    with torch.no_grad():
        clip_fraction = ((ratio - clipped).abs() > 1e-8).float()
        stats = GRPOStats(
            loss=float(total.detach()),
            policy_loss=float(policy_loss.detach()),
            kl=float((kl * mask).sum().detach() / n_tokens),
            mean_ratio=float((ratio * mask).sum().detach() / n_tokens),
            clip_fraction=float((clip_fraction * mask).sum().detach() / n_tokens),
            mean_advantage=float((advantages * mask).sum().detach() / n_tokens),
            n_tokens=n_tokens,
        )
    return total, stats


@dataclass
class TokenBatch:
    """Everything the step needs, already tokenised and aligned."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor | None = None
    advantages: torch.Tensor | None = None

    def check(self) -> None:
        if self.input_ids.shape != self.attention_mask.shape:
            raise ValueError("input_ids and attention_mask must match")
        expected = (self.input_ids.shape[0], self.input_ids.shape[1] - 1)
        if self.completion_mask.shape != self.input_ids.shape:
            raise ValueError(
                f"completion_mask must be [B, T] = {tuple(self.input_ids.shape)}, "
                f"got {tuple(self.completion_mask.shape)}"
            )
        if self.old_logprobs.shape != expected:
            raise ValueError(f"old_logprobs must be {expected}, got {tuple(self.old_logprobs.shape)}")
        if self.ref_logprobs is not None and self.ref_logprobs.shape != expected:
            raise ValueError(f"ref_logprobs must be {expected}, got {tuple(self.ref_logprobs.shape)}")
        if self.advantages is not None and self.advantages.shape != expected:
            raise ValueError(f"advantages must be {expected}, got {tuple(self.advantages.shape)}")


class GRPOTrainer:
    """One optimiser step over a batch of scored rollouts."""

    def __init__(self, model: Any, optimizer: torch.optim.Optimizer, cfg: GRPOConfig = GRPOConfig()):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg

    def step(self, batch: TokenBatch) -> GRPOStats:
        batch.check()
        advantages = self._advantages(batch)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        logits = output.logits if hasattr(output, "logits") else output
        logp = token_logprobs(logits, batch.input_ids)

        mask = batch.completion_mask[:, 1:].float()
        loss, stats = grpo_loss(
            logp, batch.old_logprobs, advantages, mask, self.cfg, batch.ref_logprobs
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], self.cfg.max_grad_norm
        )
        self.optimizer.step()
        stats.grad_norm = float(grad_norm)  # type: ignore[attr-defined]
        return stats

    def _advantages(self, batch: TokenBatch) -> torch.Tensor:
        if batch.advantages is not None:
            return batch.advantages
        raise ValueError(
            "batch has no advantages: compute them with group_advantages() from the "
            "per-rollout rewards, then expand to [B, T-1]"
        )


def expand_advantages(
    per_sample: Sequence[float], batch: TokenBatch
) -> torch.Tensor:
    """Broadcast one advantage per rollout across its completion tokens."""

    if len(per_sample) != batch.input_ids.shape[0]:
        raise ValueError("one advantage per rollout is required")
    values = torch.tensor(per_sample, dtype=torch.float32).unsqueeze(1)
    return values.expand(-1, batch.input_ids.shape[1] - 1).contiguous()


def batch_stats(batch: TokenBatch) -> Mapping[str, float]:
    """Cheap sanity numbers worth logging next to the loss."""

    mask = batch.completion_mask[:, 1:].float()
    n = float(mask.sum().item())
    return {
        "tokens": n,
        "mean_completion_len": n / max(1, batch.input_ids.shape[0]),
        "mean_old_logprob": float((batch.old_logprobs * mask).sum() / max(n, 1.0)),
    }
