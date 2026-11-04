#!/usr/bin/env bash
# First verl GRPO attempt on the 15-task lookup set.
# Everything here was discovered by probing verl 0.9.1, not guessed:
#   agent loop name   : tool_agent           (registered in tool_agent_loop.py)
#   tool loading      : multi_turn.function_tool_path -> a .py file with @function_tool
#   tool call format  : hermes               (Qwen3's native <tool_call> style)
#   dataset formats   : .jsonl accepted directly by RLHFDataset
set -u
P=/root/miniconda3/envs/myconda/bin/python
cd /root/agentic-rl-lab || exit 1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/hf
export PYTHONPATH=/root/agentic-rl-lab
export WANDB_MODE=disabled

echo "== exporting the kept task set =="
$P scripts/export_verl.py --tasks tasks/stage3-kept.json --out data/lookup-kept.jsonl --plain-system 2>&1 | tail -3

echo "== launching verl (2 steps, fail fast) =="
timeout 900 $P -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=data/lookup-kept.jsonl \
  data.val_files=data/lookup-kept.jsonl \
  data.train_batch_size=16 \
  data.val_batch_size=16 \
  data.max_prompt_length=1024 \
  data.max_response_length=384 \
  data.filter_overlong_prompts=true \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B-AWQ \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.multi_turn.function_tool_path=arlab/train/verl_tools.py \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=256 \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  custom_reward_function.path=arlab/train/verl_reward.py \
  custom_reward_function.name=compute_score \
  trainer.total_epochs=1 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.project_name=arlab \
  trainer.experiment_name=lookup-grpo-smoke \
  trainer.logger='[console]' \
  > /root/verl_run.log 2>&1
echo "EXIT=$?"
tail -25 /root/verl_run.log
