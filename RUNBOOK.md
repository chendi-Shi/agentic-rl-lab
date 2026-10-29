# 上卡手册（RUNBOOK）

第一次租卡时按顺序做，每步都有验收标准和止损点。
**任何一步没通过就不要往下走**——agentic RL 的问题会级联，越往后越难定位。

---

## 0. 本地（零卡费，必须先全绿）

```powershell
$py = "C:\anaconda\envs\dsproject\python.exe"
cd agentic-rl-lab
& $py -m pytest tests -q                    # 期望 61 passed
& $py scripts\smoke_rollout.py              # 两阶段 + resume
& $py scripts\build_verl_cmd.py --config configs\grpo_qwen3_4b.json --gpu 80
```

**验收**：测试全过；smoke 里 `resume check` 的 `run=0 skipped=6`。

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
  --config configs/grpo_qwen3_4b.json \
  --gpu 80 --hours 200 --price 7
```

**验收**：`checks: ok`；总显存估算 < 73GB。
**注意**：脚本里的 verl 参数名是模板，**第一次上卡必须对照实际版本核对** `arlab/train/launch.py` 的 `VERL_OVERRIDES`。

---

## 5. 任务接入

阶段 1（数学 + 代码解释器）已经在 `arlab/tasks/math_env.py`，导出成 `tasks/stage1.json` 即可：

```bash
python - <<'PY'
import json
from arlab.tasks.math_env import make_math_tasks
tasks = [{"task_id": t.task_id, "instruction": t.instruction,
          "expected": t.required_fields["answer"]} for t in make_math_tasks()]
open("tasks/stage1.json", "w", encoding="utf-8").write(json.dumps(tasks, ensure_ascii=False, indent=2))
PY
```

**验收**：文件非空，任务数对得上。

---

## 6. baseline（SFT 前 / SFT 后各一次）

```bash
vllm serve Qwen/Qwen3-4B --port 8000 --gpu-memory-utilization 0.55 \
  --max-model-len 2560 --max-num-seqs 8

# 另一个终端
python scripts/smoke_rollout.py   # 换成真实模型 policy 后
```

**验收**：每个任务至少能出 1 次非零分。
**止损**：如果 baseline 全 0，说明任务对 4B 太难 → 降难度，别调学习率。

---

## 7. 难度筛选（这一步决定 RL 会不会学）

```bash
python scripts/difficulty_report.py --json traces/baseline_rewards.json
```

**验收**：`keep` 里至少 30 个任务；`[OK]` 而非 `[COLLAPSED]`。
**规则**：只留通过率 20–80% 的任务。全 0 或全 1 的任务对 GRPO 是纯浪费算力。

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
# 相同任务、相同 seed，配对比较 SFT vs RL
python - <<'PY'
from arlab.eval import paired_bootstrap
# a = sft 各任务 reward, b = rl 各任务 reward（同一批任务，顺序一致）
print(paired_bootstrap(a, b).summary())
PY
```

**只报有 CI 的差值。** 不显著就写"不显著"。

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
