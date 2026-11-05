# 复现指南

两条路径：**不占卡的部分（环境、任务、诊断）**，和**需要卡的部分（训练、评测）**。

## 零、环境

```bash
git clone https://github.com/chendi-Shi/agentic-rl-lab && cd agentic-rl-lab
pip install -e ".[dev]"
pytest -q                        # 183 项测试，不需要 GPU
```

## 一、不需要卡：造题与难度标定

```bash
python scripts/make_tasks.py --stage 3            # 270 道任务 → tasks/stage3-lookup.json
python scripts/make_tasks.py --stage 1 --generated 16
```

自检（oracle 与脚本计划必须一致，否则环境本身有问题）：

```bash
pytest tests/test_lookup_tasks.py -q
```

难度标定需要 vLLM，但不需要训练：

```bash
for s in 0 1 2 3 4; do
  python scripts/run_baseline.py --tasks tasks/stage3-hard.json --policy llm \
    --base-url http://127.0.0.1:6006 --model qwen3-4b \
    --oracle answer --no-think --temperature 0.9 --seed $s \
    --store traces/cal5.jsonl
done
python scripts/filter_tasks.py --tasks tasks/stage3-hard.json --store traces/cal5.jsonl \
  --out tasks/stage3-train-band.json
```

`filter_tasks` 只保留通过率落在 20–80% 的题。全对和全错的题对 GRPO 没有任何贡献。

## 二、需要卡：训练

起服务（40G 卡能放下 4B bf16 加 LoRA 训练，关键是把 vLLM 的显存让出来）：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /root/hf/Qwen3-4B --served-model-name qwen3-4b \
  --port 6006 --max-model-len 4096 --max-num-seqs 16 \
  --gpu-memory-utilization 0.26 --enforce-eager
```

`--enforce-eager` 在这台机器的镜像上必须加（否则 inductor 报 `GraphPartitionSignature`）。
`--gpu-memory-utilization 0.26` 是给训练留显存用的；给到 0.40 再加 group_size 16 会 OOM。

```bash
python scripts/train_grpo_lookup.py \
  --tasks tasks/stage3-train-band.json \
  --model /root/hf/Qwen3-4B --served-model qwen3-4b \
  --steps 150 --group-size 8 --workers 8 --lr 3e-5 \
  --kl-coef 0.001 --lora-dropout 0.05 --temperature 0.9 --max-length 1024 \
  --tag c1 --out reports/grpo-c1.json
```

`--workers 8` 是并发 rollout，比串行快 6.7 倍（见 `scripts/bench_rollout.py`）。

训练结束会把 LoRA 存到 `checkpoints/grpo-<tag>`。

## 三、需要卡：评测与配对分析

训练完把适配器加载进 vLLM（实测 40 秒热加载）：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /root/hf/Qwen3-4B --served-model-name qwen3-4b \
  --port 6006 --max-model-len 4096 --max-num-seqs 16 \
  --gpu-memory-utilization 0.30 --enforce-eager \
  --enable-lora --max-lora-rank 16 --lora-modules trained=checkpoints/grpo-c1
```

基线与训练各跑 5 个种子（**同一批题、同一组种子**，否则没法配对）：

```bash
for seed in 0 1 2 3 4; do
  python scripts/run_baseline.py --tasks tasks/stage3-heldout.json --policy llm \
    --base-url http://127.0.0.1:6006 --model trained \
    --oracle answer --no-think --temperature 0.9 --config-id evc1 --seed $seed \
    --store traces/ev_c1.jsonl
done
python scripts/paired_eval.py       # 逐 (task, seed) 相减 + 置信区间
```

`paired_eval.py` 会读 `traces/q2_base.jsonl` 和 `traces/ev_c*.jsonl`，输出每次训练的配对差值与池化结果。

## 四、踩过的运维坑

| 现象 | 根因 | 处理 |
|---|---|---|
| `server_failed` 但训练正常 | `pkill -f vllm.entrypoints` 只杀父进程，multiprocessing worker 还占着 10 GB | 按 `nvidia-smi --query-compute-apps=pid` 逐个 kill |
| 每轮立刻 OOM，循环空转 | vLLM 0.40 + group 16 超出显存 | 降到 0.26，group 8 |
| 前台 `python -c` 被 PowerShell 拆坏 | 内联多行 Python 经 PowerShell 引号解析 | 一律写成脚本文件再 scp |
| 文件里出现 `\ufeff` | PowerShell 5.1 的 `-Encoding utf8` 会写 BOM | 上传前统一去 BOM |

## 五、数据

`reports/` 下：

- `summary2.csv` — 11 轮配置的训练与评测记录
- `paired.json` — 9 轮池化配对分析
- `rollout_bench.json` — 并发 vs 串行基准

完整轨迹与 11 个 LoRA 适配器体积较大（约 1.1 GB），未入库。
