"""Pre-flight check for the rented GPU box.

Run this before spending on a training run. It answers the three questions that
waste the most money: is CUDA visible, how much VRAM is actually free for the
rollout cache, and is the framework importable.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
from typing import Any


def _try_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        return True, str(getattr(mod, "__version__", "unknown"))
    except Exception as exc:  # noqa: BLE001 - report, never raise during preflight
        return False, f"{type(exc).__name__}: {exc}"


def collect() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "disk_free_gb": round(shutil.disk_usage(".").free / 1024**3, 1),
    }

    ok, detail = _try_import("torch")
    info["torch"] = {"available": ok, "detail": detail}
    if ok:
        import torch

        info["torch"]["cuda_built"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1),
                }
                for i in range(torch.cuda.device_count())
            ]
        else:
            info["gpus"] = []

    for pkg in ("vllm", "transformers", "peft", "trl", "verl", "rllm", "ray"):
        ok, detail = _try_import(pkg)
        info[f"pkg:{pkg}"] = {"available": ok, "detail": detail}

    return info


def verdict(info: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if not info.get("cuda_available"):
        notes.append("NO CUDA: this box cannot train; do not start a run.")
    gpus = info.get("gpus") or []
    if gpus:
        for gpu in gpus:
            if gpu["total_gb"] < 70:
                notes.append(
                    f"{gpu['name']} has only {gpu['total_gb']}GB. "
                    "7B LoRA + reference + vLLM rollout needs ~50-65GB; scale down the model or the KV budget."
                )
    if not info.get("pkg:vllm", {}).get("available"):
        notes.append("vllm missing: the rollout engine is the bottleneck of agentic RL.")
    if info.get("disk_free_gb", 0) < 100:
        notes.append(
            f"Only {info['disk_free_gb']}GB free: model weights plus trajectory logs need ~100GB."
        )
    if not notes:
        notes.append("Looks usable.")
    return notes


def main() -> int:
    info = collect()
    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"{key:<{width}}  {value}")
    print()
    for note in verdict(info):
        print(f"- {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
