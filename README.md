# agentic-rl-lab

单卡 agentic RL 的可验证骨架：**任务、奖励、评测、诊断四件事自己写，训练循环交给框架（verl / rLLM）。**

换框架不用重写，而且这四件事占了项目成败的大头——奖励设计和难度控制错了，换什么框架都救不回来。

| 模块 | 作用 | 需要 GPU |
|---|---|---|
| `arlab/tasks/` | 任务定义、可验证环境、合成业务环境、MCP 适配、录制回放 | 否 |
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
- [x] 61 项单元测试 + 两阶段 smoke + 断点续跑
- [x] 显存预检与命令模板
- [x] MCP 适配 + 录制回放（离线可调试）
- [ ] 第一次上卡：verl 示例（RUNBOOK 第 3 步）
- [ ] 阶段 1 baseline 与 GRPO 曲线
- [ ] 阶段 2 迁移与异步 partial rollout
