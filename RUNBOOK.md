# 上卡手册（RUNBOOK）

第一次租卡时按顺序做，每步都有验收标准和止损点。
**任何一步没通过就不要往下走**——agentic RL 的问题会级联，越往后越难定位。

---

## 0. 本地（零卡费，必须先全绿）

```powershell
$py = "C:\anaconda\envs\dsproject\python.exe"
cd agentic-rl-lab
& $py -m pytest tests -q                    # 期望 113 passed
& $py scripts\smoke_rollout.py              # 两阶段 + resume
& $py scripts\build_verl_cmd.py --config configs\grpo_qwen3_4b.json --gpu 80

# 全链路自检（scripted 策略，不是 baseline）
& $py scripts\make_tasks.py --stage 1
& $py scripts\run_baseline.py --tasks tasks\stage1.json --policy scripted --oracle answer
& $py scripts\run_baseline.py --tasks tasks\stage2.json --policy scripted --env mcp
& $py scripts\export_verl.py --tasks tasks\stage1.json --out data\stage1.jsonl
```

**验收**：测试全过；smoke 里 `resume check` 的 `run=0 skipped=6`；
两个 run_baseline 都是 `mean_reward≈1.07`（满分只证明管线通）。

---

## 1. 租卡

| 项 | 值 |
|---|---|
| 卡 | 1× A100-80G（不要多机） |
| 镜像 | CUDA 12.4+，Python 3.10/3.11 |
| 数据盘 | ≥ 100 GB（模型 + 轨迹日志） |
| 计价 | 按量，关机停 GPU 费 |

**先开 1 小时试机**，通过第 2 步再续。

---

## 2. 环境自检

```bash
python -m arlab.env_check
```

**验收**：`cuda_available=True`，GPU 显存 ≥ 70GB，`vllm` 可用。
**止损**：`vllm` 装不上就先别装 verl——rollout 引擎是瓶颈，装不上等于没卡。

---

## 3. 跑通框架自带的 GRPO 示例

**这一步的目的不是训练，是证明环境正确。**

```bash
git clone https://github.com/volcengine/verl
cd verl && pip install -e .
# 跑官方 example（GSM8K 即可），只跑 20 步
```

**验收**：loss 曲线有数、有 step 输出、显存不炸。
**止损**：2 天跑不通 → 换 rLLM，不要死磕。**不要在这步之前碰自己的任务。**

---

## 4. 用我们的配置预检

```bash
python scripts/build_verl_cmd.py \
  --config configs/grpo_qwen3_8b.json \
  --gpu 80 --hours 200 --price 7
```

**验收**：`checks: ok`；总显存估算 < 73GB（8B 主配置约 48GB，14B 约 60GB）。
**先用 `grpo_qwen3_4b.json`（约 25GB）把链路跑通，再换 8B**——loop 的 bug 和模型大小无关，小模型迭代快一倍。
**注意**：脚本里的 verl 参数名是模板，**第一次上卡必须对照实际版本核对** `arlab/train/launch.py` 的 `VERL_OVERRIDES`。

---

## 5. 任务接入

**6 道题培养不出策略。** GRPO 每组 8 个采样，题目全对或全错都会让 advantage 归零。
所以要生成一套有难度梯度的训练集：

```bash
python scripts/make_tasks.py --stage 1 --generated 16   # 每档 16 题 = 48 题
```

三档难度（简单算术 / 折扣均值比例 / 二次方程面积复利），每题都带可执行表达式，
生成时自动用同一个沙箱解释器验证答案 —— 模板写错会当场报错，不会变成脏奖励。

**验收**：`tiers: {'easy': 16, 'medium': 16, 'hard': 16}`，文件里 48 条。
**注意**：`tasks/stage1.json`（6 题）只用于冒烟，训练配置指向 `tasks/stage1-train.json`。

---

## 6. baseline（SFT 前 / SFT 后各一次）

```bash
# 24GB 卡（4090）：4B + AWQ
vllm serve Qwen/Qwen3-4B-AWQ --port 6006 --gpu-memory-utilization 0.85 --max-model-len 4096
# 80GB 卡：8B + AWQ
vllm serve Qwen/Qwen3-8B-AWQ --port 6006 --gpu-memory-utilization 0.85 --max-model-len 8192

# 另一个终端：真模型跑基线（--policy llm 才是 baseline）
python scripts/run_baseline.py --tasks tasks/stage1-train.json --policy llm \
  --base-url http://127.0.0.1:6006 --model Qwen/Qwen3-4B-AWQ --oracle answer \
  --config-id sft --store traces/base_sft.jsonl
```

**验收**：每个任务至少能出 1 次非零分；`tokens: in=... out=...` 不是 0（为 0 说明客户端没拿到 usage）。
**止损**：如果 baseline 全 0，说明任务对 4B 太难 → 降难度，别调学习率。

---

## 7. 难度筛选（这一步决定 RL 会不会学）

```bash
python scripts/filter_tasks.py \
  --tasks tasks/stage1-train.json \
  --store traces/base_sft.jsonl \
  --out tasks/stage1-kept.json
```

**验收**：`kept` 至少 30 个任务；`too_hard` 不应该占多数。
**规则**：只留通过率 20–80% 的任务。全对或全错的任务对 GRPO 是纯浪费算力。
**注意**：通过 = `r_task == 1.0`（完全解出），不是 `reward > 0` —— 有部分分时后者会显示全部通过。
**然后**：把训练配置的 `tasks_file` 指向 `tasks/stage1-kept.json`，用筛过的集合训练。

---

## 8. GRPO 训练

```bash
# 用第 4 步打印出来的命令
```

**训练中必须盯的三条**（前 100 步就能看出问题）：
1. `reward` 是否单调不降 —— 平的不动就是塌缩
2. 组内 reward std —— 长期为 0 说明任务难度不对
3. 平均轨迹长度 —— 暴涨说明模型学会了"多调工具混时间"

**止损**：500 步内 reward 无变化 → 停下来查第 7 步，不要加大步数。

---

## 9. 评测与出数

```bash
# 相同任务、相同 seed，配对比较 SFT vs RL（配对相同 task id）
python scripts/compare_runs.py \
  --a traces/base_sft.jsonl --b traces/rl.jsonl \
  --label-a SFT --label-b RL \
  --out reports/compare-sft-vs-rl.json
```

**只报有 CI 的差值。** `verdict` 是 `NOT significant` 就照此写，不要挑点估计讲故事。
同一条命令也能比较 SFT 前后的两版模型——只要两次运行的 task id 一致。

---

## 10. 关机前必做

- [ ] `traces/` 打包下载（这是你最贵的资产）
- [ ] 把 config、镜像 digest、verl commit 记进 `reports/`
- [ ] 把当次卡时和花费记进 `reports/cost.md`

---

## 常见故障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| OOM 在 rollout 阶段 | KV cache 撑爆 | 降 `rollout_concurrency` 或 `vllm_gpu_util` |
| reward 全程 0 | 任务太难 / 奖励 bug | 先跑 `difficulty_report`，再查奖励 |
| 组内 std 恒 0 | 任务全对或全错 | 换任务，别调参 |
| 训练极慢 | 长尾轨迹拖 batch | 降 `max_steps`，上 partial rollout |
| 工具超时 | 沙箱/网络抖动 | 工具幂等 + 固定超时 + 结果缓存 |
