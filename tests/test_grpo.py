from __future__ import annotations

import math

import pytest
import torch

from arlab.train.grpo import (
    GRPOConfig,
    GRPOTrainer,
    TokenBatch,
    expand_advantages,
    group_advantages,
    grpo_loss,
    token_logprobs,
)


class ToyLM(torch.nn.Module):
    """Smallest thing that satisfies the trainer's duck-typed contract."""

    def __init__(self, vocab: int = 11, hidden: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        return self.head(torch.tanh(self.embed(input_ids)))


# --------------------------------------------------------------- advantages


def test_advantages_are_normalised_per_group():
    rewards = [1.0, 0.0, 1.0, 0.0]
    groups = ["a", "a", "b", "b"]

    adv = group_advantages(rewards, groups)

    for group in ("a", "b"):
        values = [adv[i] for i, g in enumerate(groups) if g == group]
        assert sum(values) == pytest.approx(0.0, abs=1e-5)
        # std of two samples {+x, -x} is x, so the normalised values are +-1
        assert values[0] == pytest.approx(1.0, abs=1e-4)
        assert values[1] == pytest.approx(-1.0, abs=1e-4)


def test_identical_rewards_collapse_to_zero():
    """The all-solved / all-failed case: no signal, and it must be visible."""
    adv = group_advantages([1.0, 1.0, 1.0], ["a", "a", "a"])
    assert adv == [0.0, 0.0, 0.0]


def test_single_sample_group_gets_no_advantage():
    assert group_advantages([0.7], ["a"]) == [0.0]


def test_advantages_reject_mismatched_length():
    with pytest.raises(ValueError):
        group_advantages([1.0, 2.0], ["a"])
    with pytest.raises(ValueError):
        group_advantages([], [])


# ------------------------------------------------------------------ logprob


def test_token_logprobs_picks_the_observed_tokens():
    logits = torch.zeros(1, 4, 5)  # uniform -> log 1/5 everywhere
    ids = torch.tensor([[0, 1, 2, 3]])

    logp = token_logprobs(logits, ids)

    assert logp.shape == (1, 3)
    assert torch.allclose(logp, torch.full((1, 3), math.log(1 / 5)), atol=1e-6)


def test_token_logprobs_rejects_bad_shapes():
    with pytest.raises(ValueError):
        token_logprobs(torch.zeros(4, 5), torch.tensor([[0, 1]]))
    with pytest.raises(ValueError):
        token_logprobs(torch.zeros(1, 4, 5), torch.zeros(2, 4, dtype=torch.long))


# --------------------------------------------------------------------- loss


def _tensors(B: int = 2, T: int = 6) -> tuple[torch.Tensor, ...]:
    logp = torch.full((B, T - 1), -1.0)
    old = torch.full((B, T - 1), -1.0)
    mask = torch.ones(B, T - 1)
    adv = torch.zeros(B, T - 1)
    return logp, old, mask, adv


def test_ratio_one_reduces_to_reinforce():
    logp, old, mask, adv = _tensors()
    adv[0] = 1.0
    adv[1] = -1.0

    loss, stats = grpo_loss(logp, old, adv, mask, GRPOConfig())

    assert stats.mean_ratio == pytest.approx(1.0)
    assert stats.clip_fraction == pytest.approx(0.0)
    # -(mean advantage) = -((+1) + (-1))/2 = 0
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert stats.n_tokens == 10


def test_clipping_caps_the_surrogate():
    logp, old, mask, adv = _tensors()
    old = old - 2.0  # ratio = exp(2) ~ 7.39
    adv[:] = 1.0

    _, stats = grpo_loss(logp, old, adv, mask, GRPOConfig(clip=0.2))

    assert stats.mean_ratio == pytest.approx(math.exp(2.0), rel=1e-4)
    assert stats.clip_fraction == pytest.approx(1.0)
    # clipped surrogate is 1.2, so policy loss is -1.2
    assert stats.policy_loss == pytest.approx(-1.2, rel=1e-4)


def test_negative_advantage_is_clipped_on_the_other_side():
    logp, old, mask, adv = _tensors()
    old = old - 2.0  # ratio 7.39, advantage -1
    adv[:] = -1.0

    _, stats = grpo_loss(logp, old, adv, mask, GRPOConfig(clip=0.2))

    # min(7.39 * -1, 1.2 * -1) = -7.39 -> policy loss +7.39 (the pessimistic branch)
    assert stats.policy_loss == pytest.approx(math.exp(2.0), rel=1e-3)


def test_kl_is_zero_when_reference_matches_policy():
    logp, old, mask, adv = _tensors()

    _, stats = grpo_loss(logp, old, adv, mask, GRPOConfig(), ref_logp=logp.clone())

    assert stats.kl == pytest.approx(0.0, abs=1e-7)


def test_kl_k3_is_non_negative_and_matches_the_formula():
    logp, old, mask, adv = _tensors()
    ref = logp + 0.5

    _, stats = grpo_loss(logp, old, adv, mask, GRPOConfig(), ref_logp=ref)

    expected = math.exp(0.5) - 0.5 - 1.0
    assert stats.kl == pytest.approx(expected, rel=1e-5)
    assert stats.kl > 0


def test_kl_weight_scales_the_total():
    logp, old, mask, adv = _tensors()
    ref = logp + 0.5
    kl = math.exp(0.5) - 0.5 - 1.0

    _, no_kl = grpo_loss(logp, old, adv, mask, GRPOConfig(kl_coef=0.0), ref_logp=ref)
    _, with_kl = grpo_loss(logp, old, adv, mask, GRPOConfig(kl_coef=1.0), ref_logp=ref)

    assert with_kl.loss - no_kl.loss == pytest.approx(kl, rel=1e-5)


def test_empty_completion_is_an_error():
    logp, old, mask, adv = _tensors()
    with pytest.raises(ValueError, match="no completion tokens"):
        grpo_loss(logp, old, adv, torch.zeros_like(mask), GRPOConfig())


def test_shape_mismatches_are_rejected():
    logp, old, mask, adv = _tensors()
    with pytest.raises(ValueError):
        grpo_loss(logp, old[:-1], adv, mask, GRPOConfig())
    with pytest.raises(ValueError):
        grpo_loss(logp, old, adv, mask[:-1], GRPOConfig())


# -------------------------------------------------------------------- batch


def make_batch(
    model: ToyLM | None = None,
    B: int = 2,
    T: int = 6,
    prompt: int = 3,
    vocab: int = 11,
) -> TokenBatch:
    """Build a batch whose ``old_logprobs`` come from **the model under training**.

    Using a separately seeded model here would make ratio != 1 and silently
    invalidate every clip/ratio assertion below.
    """

    if model is None:
        torch.manual_seed(0)
        model = ToyLM(vocab)
    ids = torch.randint(0, vocab, (B, T), generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        logits = model(ids)
    logp = token_logprobs(logits, ids)
    mask = torch.zeros(B, T)
    mask[:, prompt:] = 1.0
    return TokenBatch(
        input_ids=ids,
        attention_mask=torch.ones(B, T, dtype=torch.long),
        completion_mask=mask,
        old_logprobs=logp.detach(),
    )


def test_batch_validation():
    batch = make_batch()
    batch.check()

    bad = make_batch()
    bad.completion_mask = bad.completion_mask[:, :-1]
    with pytest.raises(ValueError):
        bad.check()


def test_expand_advantages_broadcasts_per_rollout():
    batch = make_batch()
    adv = expand_advantages([0.5, -0.5], batch)

    assert adv.shape == (2, batch.input_ids.shape[1] - 1)
    assert torch.allclose(adv[0], torch.full_like(adv[0], 0.5))


def test_expand_advantages_requires_one_per_rollout():
    batch = make_batch()
    with pytest.raises(ValueError):
        expand_advantages([1.0], batch)


# --------------------------------------------------------------- full step


def test_one_step_moves_the_parameters():
    torch.manual_seed(1)
    model = ToyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = GRPOTrainer(model, optimizer, GRPOConfig())

    batch = make_batch(model)
    batch.advantages = expand_advantages([1.0, -1.0], batch)
    before = [p.detach().clone() for p in model.parameters()]

    stats = trainer.step(batch)

    assert math.isfinite(stats.loss)
    assert stats.mean_ratio == pytest.approx(1.0, abs=1e-3)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_step_requires_advantages():
    model = ToyLM()
    trainer = GRPOTrainer(model, torch.optim.SGD(model.parameters(), lr=0.1))

    with pytest.raises(ValueError, match="no advantages"):
        trainer.step(make_batch())


def test_collapsed_group_produces_no_policy_gradient():
    """All-equal rewards must yield a zero gradient, and the stats must show it."""
    torch.manual_seed(2)
    model = ToyLM()
    trainer = GRPOTrainer(model, torch.optim.SGD(model.parameters(), lr=0.1))
    batch = make_batch(model)
    advantages = group_advantages([0.0, 0.0], ["same", "same"])
    batch.advantages = expand_advantages(advantages, batch)

    stats = trainer.step(batch)

    assert stats.mean_advantage == pytest.approx(0.0)
    assert stats.policy_loss == pytest.approx(0.0, abs=1e-8)
