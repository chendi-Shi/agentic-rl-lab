#!/usr/bin/env bash
# Install a verl-compatible stack from scratch, with the two traps this box has.
#
# Why these exact versions:
#
#   verl 0.9.1        needs huggingface-hub >= 1.0 (it imports BucketNotFoundError)
#   hub >= 1.0        needs transformers 5.x
#   transformers 5.x  needs vllm >= 0.13
#   vllm >= 0.13      needs torch 2.9
#
#   vllm 0.8.5        needs transformers < 4.52 -> hub < 1.0   (incompatible)
#
# so the working combination is torch 2.9 + transformers 5.x + vllm >= 0.13 +
# verl 0.9.1. The driver on this image caps CUDA at 12.8, which cu128 wheels
# satisfy.
#
# Traps:
#   1. download.pytorch.org stalls (the pip cache sat at 2985 MB for minutes).
#      Aliyun mirrors the same wheels and answers in seconds.
#   2. The image's pip.conf points at a tuna index that returns 403 for /simple/.
#      Force pypi.org.
#
# Usage:  bash scripts/setup_verl_stack.sh
# Exits non-zero if verl.trainer.main_ppo cannot be imported.
set -u
P=${PYTHON:-/root/miniconda3/envs/myconda/bin/python}
export PIP_INDEX_URL=https://pypi.org/simple
TORCH_MIRROR=https://mirrors.aliyun.com/pytorch-wheels/cu128/

echo "== 1. torch 2.9 cu128 (from the mirror, ~5 min) =="
$P -m pip install "torch==2.9.0" -f "$TORCH_MIRROR" 2>&1 | tail -3
$P -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || exit 1

echo "== 2. vllm + transformers + hub + verl =="
$P -m pip install --no-cache-dir -U vllm transformers huggingface-hub verl 2>&1 | tail -4

echo "== 3. verify =="
$P - <<'PY'
import importlib
checks = [
    ("torch", lambda: (lambda t: f"{t.__version__} cuda={t.cuda.is_available()}")(importlib.import_module("torch"))),
    ("transformers", lambda: importlib.import_module("transformers").__version__),
    ("huggingface-hub", lambda: importlib.import_module("huggingface_hub").__version__),
    ("vllm", lambda: importlib.import_module("vllm").__version__),
    ("verl", lambda: getattr(importlib.import_module("verl"), "__version__", "?")),
    ("verl.trainer.main_ppo", lambda: "ok" if importlib.import_module("verl.trainer.main_ppo") else "?"),
]
failed = []
for name, fn in checks:
    try:
        print(f"  {name}: OK -> {fn()}")
    except Exception as exc:
        failed.append(name)
        print(f"  {name}: FAILS -> {type(exc).__name__}: {str(exc)[:110]}")
raise SystemExit(1 if "verl.trainer.main_ppo" in failed else 0)
PY
rc=$?

if [ $rc -ne 0 ]; then
  echo
  echo "verl still cannot import. Do not keep installing -- the fallback trainer in"
  echo "arlab/train/grpo.py produced every result in this repo and needs no verl."
  echo "Run the experiments with it instead of burning the GPU-day on installs."
  exit $rc
fi

echo
echo "== 4. register our tools with verl and check the schemas =="
cd "$(dirname "$0")/.." || exit 1
PYTHONPATH=. $P - <<'PY'
import arlab.train.verl_tools as tools
from verl.tools.function_tool import FUNCTION_TOOL_REGISTRY as registry
print("registered:", sorted(registry))
for name in tools.TOOL_NAMES:
    schema = registry[name].tool_schema.model_dump(exclude_unset=True, exclude_none=True)
    print(name, "->", schema["function"]["parameters"]["properties"])
print("call:", tools.orders_get("A123"))
PY
