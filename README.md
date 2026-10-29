# agentic-rl-lab

[![ci](https://github.com/chendi-Shi/agentic-rl-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/chendi-Shi/agentic-rl-lab/actions/workflows/ci.yml)

单卡 agentic RL 的可验证骨架：**任务、奖励、评测、诊断四件事自己写，训练循环交给框架（verl / rLLM）。**

换框架不用重写，而且这四件事占了项目成败的大头——奖励设计和难度控制错了，换什么框架都救不回来。

| 模块 | 作用 | 需要 GPU |
|---|---|---|
| `arlab/tasks/` | 任务定义、可验证环境、合成业务环境、MCP 适配、录制回放 | 否 |
| `arlab/policy/` | OpenAI 兼容客户端（仅标准库）+ 动作解析 + 上下文遮罩 | 否 |
| `arlab/reward/` | 四段式状态奖励 + 答案奖励（含结构化 JSON 部分分） | 否 |
| `arlab/rollout/` | 轨迹循环、步数预算、断点续跑 | 否 |
| `arlab/store/` | JSONL 轨迹存储，按 `(task, policy, config, seed)` 去重 | 否 |
| `arlab/diagnostics/` | 优势塌缩、难度分档、长度长尾 | 否 |
| `arlab/eval/` | 配对比较 + bootstrap 置信区间 | 否 |
| `arlab/train/` | 训练配置、显存预检、verl/vLLM 命令模板 | 否 |
| `arlab/env_check.py` | 上卡前自检 | 是 |

## 为什么先写这层

agentic RL 不收敛，九成不是框架问题，是这三件事：

1. **奖励**：稀疏二值奖励 → 整组全零 → 优势塌缩 → 梯度为零 → "训练在跑但不学"
2. **难度**：任务太难或太易，组内方差为 0，GRPO 的分组优势失效
3. **长尾**：一条 30 步轨迹拖慢整个 batch，吞吐崩掉，被误判成算法问题

三件事全都能在 CPU 上测出来，一行卡都不用租。

## Infra：长尾轨迹调度

`arlab/rollout/schedule_sim.py` 是离线调度模型，**不需要卡**就能量出"长尾吃掉多少吞吐"。

```powershell
& $py scripts\scheduler_bench.py --episodes 128 --concurrency 8 --step-ms 320
```

128 条重尾轨迹（中位 4 步、最长 40 步），8 并发，每步 320ms（8B 模型 + Docker 工具）：

| | 墙钟 | 槽位利用率 | 空闲槽位时间 |
|---|---:|---:|---:|
| 同步 batch | 88.3 s | 35.4% | 456.6 s |
| 异步补位 | 38.4 s | 81.4% | 57.3 s |

**2.30 倍，纯调度收益，算法一行没改。**

同一批轨迹，round budget = 8 步时：

| | 完成 | 丢弃 | 浪费步数 |
|---|---:|---:|---:|
| 截断 | 110/128 | 18 | 24.7% |
| partial rollout | 128/128 | 0 | 0% |

丢的不只是算力——**截断等于把长任务记成失败**，奖励会被系统性偏向"短任务"，模型学会的是"别做长活"。

> 这是模型估算，不是实测：`--step-ms` 上卡后换成实测值，比例依然成立。
> 同一接口也可以接真实延迟分布，不需要改调度逻辑。

## 上卡前就能跑通的全链路

六个脚本，全部不需要 GPU：

```powershell
$py = "C:\anaconda\envs\dsproject\python.exe"

# 1. 冻结任务集（阶段 2 从 services.json 生成，零人工标注）
& $py scripts\make_tasks.py --stage 1
& $py scripts\make_tasks.py --stage 2 --services ..\byte-agent-platform\examples\corpus\services.json

# 2. 跑一遍（--policy scripted 是自检，不是 baseline）
& $py scripts\run_baseline.py --tasks tasks\stage1.json --policy scripted --oracle answer
& $py scripts\run_baseline.py --tasks tasks\stage2.json --policy scripted --env mcp   # 走真实 stdio MCP

# 3. 导成 verl 能吃的格式（写完即校验 schema）
& $py scripts\export_verl.py --tasks tasks\stage1.json --out data\stage1.jsonl

# 4. 两次运行配对比较（只认带 CI 的差值）
& $py scripts\compare_runs.py --a traces\sft.jsonl --b traces\rl.jsonl --label-a SFT --label-b RL

# 5. 换真模型时只改这一处
& $py scripts\run_baseline.py --tasks tasks\stage1.json --policy llm `
    --base-url http://127.0.0.1:8000 --model qwen3-8b --oracle answer
```

`--policy scripted` 会重放正确动作，所以它得满分只证明**管线通**，绝不能当模型成绩报。
`--policy llm` 才是真基线。

## 快速开始

```powershell
$py = "C:\anaconda\envs\dsproject\python.exe"
cd agentic-rl-lab

& $py -m pytest tests -q                              # 61 passed
& $py scripts\smoke_rollout.py                        # 两阶段 + 断点续跑
& $py scripts\replay_demo.py                          # 奖励分项演示
& $py scripts\difficulty_report.py --demo             # 难度分档 + 塌缩检测
& $py scripts\build_verl_cmd.py --config configs\grpo_qwen3_4b.json --gpu 80 --hours 200 --price 7
& $py -m arlab.env_check                              # 上卡前自检
```

## 两个任务阶段

**阶段 1（证明 loop 是对的）**：`arlab/tasks/math_env.py`
数学题 + 代码解释器，答案精确匹配。公开证据表明 4B–7B 用 GRPO 能涨，先复现出来。

**阶段 2（原创）**：`tasks/stage2.json`
从 `byte-agent-platform` 的 32 个服务任务离线生成（24 dev / 8 holdout），
标签直接来自 `services.json`，**零人工标注**。判分按 6 个字段逐项给部分分
（`service / error_rate / status / likely_cause / recommendation / uncertainty`）；
`citations` 故意不参与离线判分——它只能对着 MCP 服务的实时证据 id 校验。

```powershell
& $py scripts\make_tasks.py --stage 1
& $py scripts\make_tasks.py --stage 2 --services ..\byte-agent-platform\examples\corpus\services.json
```

## 奖励设计要点

- `r_task` **只读环境状态或提交的答案**，模型自述"我退款了"不得分
- 部分分是稠密的：状态任务按必需字段数，结构化任务按字段加权
- 格式奖励前 3 步权重 1.0，之后 0.3 衰减，避免语法挤占任务信号
- 重复调用、非法工具、超预算都有明确惩罚

## 训练侧预检（不用卡）

```powershell
& $py scripts\build_verl_cmd.py --config configs\grpo_qwen3_4b.json --gpu 80 --hours 200 --price 7
```

输出显存分项估算、校验项、verl 与 vLLM 命令、成本。
4B / 8B / 24GB / 80GB 都能直接对比着算。

| 配置 | 估算峰值 | 单卡 80GB | 用途 |
|---|---:|---|---|
| `grpo_qwen3_4b.json` | 24.8 GB | 宽松 | 快速迭代、验证 loop |
| `grpo_qwen3_8b.json` | 48.1 GB | **舒适（主配置）** | 正式跑 |
| `grpo_qwen3_14b_awq.json` | 60.1 GB | 可用 | 更强基座 |
| 32B（4bit 训练 + 4bit rollout） | ~98 GB | 不够 | 需要多卡 |

估算里四份常驻开销必须同时存在：训练权重、KL 参考模型、vLLM 的**另一份**权重、KV cache。
只算权重会得出"80GB 能跑 32B 训练"的错误结论——那只是在算推理。

> `arlab/train/launch.py` 里的 `VERL_OVERRIDES` 是**模板**：verl 的参数名随版本变化，
> 第一次上卡必须对照实际版本核对这一处，其余代码不用动。

## 目录

```
arlab/
├── tasks/        schema · env · business_tasks · math_env · mcp_env · replay_env
├── reward/       verifiable（状态） · answer（答案 / 结构化 JSON）
├── rollout/      runner（预算、续跑、奖励）
├── store/        trace（JSONL、去重）
├── diagnostics/  stats（塌缩、难度、长尾）
├── eval/         harness（配对 bootstrap）
└── train/        config（显存预检） · launch（命令模板）
scripts/          smoke · replay · difficulty · build_verl_cmd
configs/          grpo_qwen3_4b.json · grpo_qwen3_8b.json
```

## 上卡

看 [RUNBOOK.md](RUNBOOK.md)。十步，每步有验收标准和止损点。

## 当前状态

- [x] 任务/奖励/评测/诊断/存储/rollout 全部落地，CPU 可测
- [x] 113 项单元测试 + 两阶段 smoke + 断点续跑
- [x] 显存预检与命令模板
- [x] MCP 适配（真 stdio 传输）+ 录制回放
- [x] 长尾调度模型：异步补位 / partial rollout 的收益量化
- [x] partial rollout 执行器：跨轮续跑、零重算
- [x] 模型客户端（OpenAI 兼容，仅标准库）+ 上下文遮罩
- [x] 轨迹 → verl 数据集导出（写完即校验）
- [x] 配对比较脚本（bootstrap CI）
- [ ] 第一次上卡：verl 示例（RUNBOOK 第 3 步）
- [ ] 阶段 1 baseline 与 GRPO 曲线
- [ ] 把调度模型接到真实 rollout（用实测延迟替换模拟值）
