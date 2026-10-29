"""Which model / precision combinations fit a given card.

    python scripts/card_fit.py --gpu 24     # 4090 / 3090
    python scripts/card_fit.py --gpu 80     # A100 / H100

Answers the question that decides the rental bill: on this card, how big a model
can I do **GRPO** on -- not inference, which is a different and much cheaper
question because training also holds the reference model and a second copy of the
weights for the rollout engine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arlab.train import TrainConfig, estimate_vram_gb  # noqa: E402

VARIANTS: list[tuple[str, float, dict]] = [
    ("Qwen3-4B", 4.0, {}),
    ("Qwen3-4B + 4bit rollout", 4.0, {"rollout_4bit": True}),
    ("Qwen3-4B 4bit train + 4bit rollout", 4.0, {"train_4bit": True, "rollout_4bit": True}),
    ("Qwen3-8B", 8.0, {}),
    ("Qwen3-8B + 4bit rollout", 8.0, {"rollout_4bit": True}),
    ("Qwen3-8B 4bit train + 4bit rollout", 8.0, {"train_4bit": True, "rollout_4bit": True}),
    ("Qwen3-14B + 4bit rollout", 14.0, {"rollout_4bit": True}),
    ("Qwen3-32B 4bit train + 4bit rollout", 32.0, {"train_4bit": True, "rollout_4bit": True}),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=float, required=True, help="VRAM in GB")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    usable = args.gpu * 0.92
    print(f"card {args.gpu:.0f}GB  ->  usable ~{usable:.0f}GB   (rollout concurrency {args.concurrency})")
    print(f"{'variant':<38}{'peak':>9}{'KV':>8}   verdict")
    print("-" * 74)

    for name, params, overrides in VARIANTS:
        cfg = TrainConfig(params_b=params, rollout_concurrency=args.concurrency, **overrides)
        estimate = estimate_vram_gb(cfg)
        fits = estimate["total"] <= usable
        print(
            f"{name:<38}{estimate['total']:>8.1f}G{estimate['kv_cache']:>7.1f}G   "
            f"{'fits' if fits else 'NO'}"
        )

    print()
    print("Notes: four copies are resident at once (train weights, KL reference,")
    print("a separate vLLM copy, KV cache). Inference-only math is much smaller.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
