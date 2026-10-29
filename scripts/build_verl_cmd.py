"""Pre-flight for the GPU day: VRAM estimate, commands, cost.

    python scripts/build_verl_cmd.py --config configs/grpo_qwen3_4b.json
    python scripts/build_verl_cmd.py --config configs/grpo_qwen3_8b.json --gpu 80 --hours 200 --price 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arlab.train import TrainConfig, dry_run, estimate_cost, preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=float, default=80.0, help="GPU memory in GB")
    parser.add_argument("--hours", type=float, default=0.0)
    parser.add_argument("--price", type=float, default=0.0, help="currency per GPU-hour")
    args = parser.parse_args()

    cfg = TrainConfig.from_json(args.config)
    report = dry_run(cfg, gpu_gb=args.gpu)

    print("=" * 74)
    print(f"config      {args.config}")
    print(f"model       {cfg.model}  (~{cfg.params_b}B params)")
    print(f"gpu budget  {args.gpu:.0f} GB")
    print("=" * 74)

    print("\nVRAM estimate")
    for name, value in report["vram"].items():
        marker = "  <-- total" if name == "total" else ""
        print(f"  {name:<20}{value:>8.2f} GB{marker}")

    warnings = preflight(cfg, gpu_gb=args.gpu)
    print("\nchecks")
    if warnings:
        for warning in warnings:
            print(f"  ! {warning}")
    else:
        print("  ok")

    print("\ntrainer")
    print(f"  {report['verl_command']}")
    print("\neval server")
    print(f"  {report['vllm_command']}")

    if args.hours and args.price:
        cost = estimate_cost(cfg, args.hours, args.price)
        print(f"\ncost\n  {cost.hours:.0f} h x {cost.hourly_price:.2f} = {cost.total:.2f}")

    out = Path("reports") / f"preflight-{cfg.experiment}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
