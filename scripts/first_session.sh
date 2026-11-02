#!/usr/bin/env bash
# First session on a rented GPU box. Checks the machine, installs the CPU-side
# deps, and tells you the exact next command. It does NOT start a server for you:
# background processes that outlive a script are a good way to burn money.
#
#   bash scripts/first_session.sh
set -euo pipefail

echo "== 1/6 GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "== 2/6 python =="
python -V

echo
echo "== 3/6 deps =="
python -m pip install -q --upgrade pip
python -m pip install -q numpy pytest

echo
echo "== 4/6 tests =="
python -m pytest tests -q

echo
echo "== 5/6 task sets =="
python scripts/make_tasks.py --stage 1

echo
echo "== 6/6 vllm =="
if python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null; then
  :
else
  echo "vllm not installed.  pip install vllm"
fi

cat <<'TXT'

--------------------------------------------------------------
Next, in this order:

1) start the server (pick the model your card can hold):
     export HF_ENDPOINT=https://hf-mirror.com
     # 24GB card:
     vllm serve Qwen/Qwen3-4B-AWQ  --port 6006 --gpu-memory-utilization 0.85 --max-model-len 4096
     # 80GB card:
     vllm serve Qwen/Qwen3-8B-AWQ  --port 6006 --gpu-memory-utilization 0.85 --max-model-len 8192

2) wait until it answers, do not guess:
     curl -s http://127.0.0.1:6006/v1/models

3) measure the baseline:
     python scripts/run_baseline.py --tasks tasks/stage1.json \
       --policy llm --base-url http://127.0.0.1:6006 \
       --model <same-model-name> --oracle answer

4) only after that, touch verl.

Shut the box down when you stop: billing runs from power-on, not from GPU use.
--------------------------------------------------------------
TXT
