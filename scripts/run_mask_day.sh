#!/usr/bin/env bash
# GPU-day runbook: mask ablation, then the KL-vs-steps sweep, then the partial
# rollout benchmark. In that order, because the first one can invalidate the other
# two: if the corrected mask is what keeps KL flat, then the nine coarse-mask runs
# in reports/ need to be re-read in that light.
#
# Every block is time-boxed and every block survives losing the GPU to a dead
# worker: the first thing each server start does is kill whatever the driver says
# is still holding memory. On 2026-11-04 a leaked vLLM worker held 10.5 GB and the
# queue spun through eighteen failed rounds in twenty minutes because nothing
# checked.
#
# Usage:  bash scripts/run_mask_day.sh 2>&1 | tee /root/maskday.log
set -u
P=${PYTHON:-/root/miniconda3/envs/myconda/bin/python}
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=$PWD
export HF_HOME=${HF_HOME:-/root/hf}
export TOKENIZERS_PARALLELISM=false
MODEL=${MODEL:-/root/hf/Qwen3-4B}
VLLM_UTIL=${VLLM_UTIL:-0.26}

cleanup_gpu() {
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  for pid in $pids; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
  done
  pkill -9 -f "[v]llm.entrypoints" 2>/dev/null
  pkill -9 -f "spawn_main" 2>/dev/null
  sleep 8
  echo "  gpu after cleanup: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

start_server() {
  cleanup_gpu
  nohup $P -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen3-4b \
    --host 127.0.0.1 --port 6006 --max-model-len 4096 --max-num-seqs 16 \
    --gpu-memory-utilization "$VLLM_UTIL" --enforce-eager "$@" \
    > /root/vllm_maskday.log 2>&1 &
  for _ in $(seq 1 60); do
    curl -s --max-time 2 http://127.0.0.1:6006/v1/models >/dev/null && return 0
    sleep 5
  done
  echo "  server FAILED to start"
  return 1
}

eval_seeds() {
  local tag=$1 count=${2:-5}
  rm -f "traces/ev_$tag.jsonl"
  for seed in $(seq 0 $((count - 1))); do
    $P scripts/run_baseline.py --tasks tasks/stage3-heldout.json --policy llm \
      --base-url http://127.0.0.1:6006 --model trained \
      --oracle answer --no-think --temperature 0.9 --config-id "ev$tag" --seed "$seed" \
      --store "traces/ev_$tag.jsonl" 2>&1 | grep -E "^tasks=" || true
  done
}

echo "=========== block 1: mask ablation (150 steps, lr 3e-5) ==========="
echo "coarse-mask reference: reports/grpo-c1.json .. c9 (KL 0.007 -> 10 over 200 steps)"
start_server || exit 1

rm -f traces/tr_m1.jsonl
timeout 2700 $P scripts/train_grpo_lookup.py \
  --tasks tasks/stage3-train-band.json --model "$MODEL" --served-model qwen3-4b \
  --steps 150 --group-size 8 --workers 8 --lr 3e-5 \
  --kl-coef 0.001 --lora-dropout 0.05 --temperature 0.9 --max-length 1024 \
  --tag m1 --store traces/tr_m1.jsonl --out reports/grpo-m1.json
echo "train exit=$?"
$P scripts/kl_trace.py reports/grpo-m1.json reports/grpo-c1.json

if [ -d checkpoints/grpo-m1 ]; then
  if start_server --enable-lora --max-lora-rank 16 --lora-modules trained=checkpoints/grpo-m1; then
    eval_seeds m1 5
    $P scripts/eval_pair.py traces/ev_m1.jsonl | head -1
    $P scripts/paired_eval.py | tail -8
  fi
else
  echo "  no adapter saved; skipping evaluation"
fi

echo
echo "=========== block 2: KL vs steps (no eval, the curve is the output) ==========="
for steps in 300 450; do
  start_server || break
  rm -f "traces/tr_m$steps.jsonl"
  timeout 5400 $P scripts/train_grpo_lookup.py \
    --tasks tasks/stage3-train-band.json --model "$MODEL" --served-model qwen3-4b \
    --steps "$steps" --group-size 8 --workers 8 --lr 3e-5 \
    --kl-coef 0.001 --lora-dropout 0.05 --temperature 0.9 --max-length 1024 \
    --tag "m$steps" --store "traces/tr_m$steps.jsonl" --out "reports/grpo-m$steps.json"
  echo "train exit=$? for $steps steps"
done
$P scripts/kl_trace.py reports/grpo-m150.json reports/grpo-m300.json reports/grpo-m450.json 2>/dev/null \
  || $P scripts/kl_trace.py reports/grpo-m1.json reports/grpo-m300.json reports/grpo-m450.json

echo
echo "=========== block 3: partial rollout on real episodes ==========="
if start_server; then
  timeout 1800 $P scripts/bench_partial.py --task agg-sum-all --group 8 \
    --model qwen3-4b --out reports/partial_bench.json
fi

echo
echo "=========== collect ==========="
mkdir -p /root/handoff
cp -r reports traces checkpoints /root/handoff/ 2>/dev/null
tar -czf /root/handoff.tar.gz -C /root handoff 2>/dev/null
ls -lh /root/handoff.tar.gz
echo "MASKDAY_DONE"
