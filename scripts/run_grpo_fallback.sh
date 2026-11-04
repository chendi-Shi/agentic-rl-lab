#!/usr/bin/env bash
# Fallback training path: vLLM serves Qwen3-4B (bf16, same weights the trainer
# loads), this script runs the self-contained GRPO loop.
set -u
P=/root/miniconda3/envs/myconda/bin/python
cd /root/agentic-rl-lab || exit 1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/hf

echo "== 1. model =="
$P - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("Qwen/Qwen3-4B", local_dir="/root/hf/Qwen3-4B",
                         allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("model at", path)
PY

echo "== 2. vllm serving the same weights =="
pkill -f "[v]llm.entrypoints" 2>/dev/null
sleep 3
nohup $P -m vllm.entrypoints.openai.api_server \
  --model /root/hf/Qwen3-4B --served-model-name qwen3-4b \
  --host 127.0.0.1 --port 6006 \
  --max-model-len 4096 --max-num-seqs 8 \
  --gpu-memory-utilization 0.40 --enforce-eager \
  > /root/vllm_bf16.log 2>&1 &

for i in $(seq 1 60); do
  if curl -s --max-time 2 http://127.0.0.1:6006/v1/models >/dev/null; then
    echo "server READY after $((i*5))s"
    break
  fi
  sleep 5
done

echo "== 3. grpo =="
$P scripts/train_grpo_lookup.py \
  --tasks tasks/stage3-kept.json \
  --model /root/hf/Qwen3-4B \
  --served-model qwen3-4b \
  --steps 12 --group-size 4 --temperature 0.9 \
  --store traces/grpo-train.jsonl \
  --out reports/grpo-lookup.json
echo "TRAIN_EXIT=$?"
