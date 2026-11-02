# verl 接入方案

> 主路径：**verl 跑 GRPO**。自研的 `arlab/train/grpo.py` 保留为**参考实现**（数值对照 + verl 不可用时的兜底）。
> 本文只写"改回 verl 之后要做什么"，不重复离线部分。

---

## 1. 三个挂载点，两个已经做完

| 挂载点 | 状态 | 位置 |
|---|---|---|
| 训练数据 | ✅ 已完成，schema 已校验 | `scripts/export_verl.py` → `data/*.jsonl` |
| 奖励函数 | ✅ 已完成，纯 Python，无 torch 依赖 | `arlab/train/verl_reward.py::compute_score` |
| **多轮环境循环** | ❌ **唯一未知** | 见第 3 节 |

奖励函数按 verl 的约定签名实现，可以直接被 import：

```python
compute_score(data_source, solution_str, ground_truth, extra_info) -> {"score": float, "acc": float}
```

挂载方式是配置，不是改代码：

```bash
custom_reward_function.path=arlab/train/verl_reward.py
custom_reward_function.name=compute_score
```

---

## 2. 上卡第一步：先确认版本，再决定怎么接

**不要先写代码。** 先花 5 分钟问清楚装上的 verl 长什么样：

```bash
pip install verl
python -c "import verl; print('version', getattr(verl, '__version__', 'unknown'))"
python -c "import verl, pkgutil; print([m.name for m in pkgutil.iter_modules(verl.__path__)])"
python -c "import verl; print(verl.__file__)"
```

然后用这三条判断走哪条路：

| 观察 | 走哪条路 |
|---|---|
| 有 `agent` / `agent_loop` / `experimental` 相关子模块 | **路径 A**：用官方多轮接口，写 30–50 行 adapter |
| 只有单轮 `rollout`（prompt → response） | **路径 B**：把我们的 runner 包成 rollout wrapper，100–150 行 |
| 接口要求改框架源码才能跑多轮 | **止损**：用自研训练步，别改别人源码 |

把这三条的输出贴出来，路径就定了，不用猜。

---

## 3. 两条路的具体形状

### 路径 A：verl 自带多轮

需要写的只有"环境适配器"：给定一条 prompt，跑完整个 agent 循环，返回轨迹。
我们这边的零件已经齐了：

- `arlab/rollout/runner.py` —— 循环、预算、工具错误隔离
- `arlab/tasks/*` —— 环境（MathEnv / SyntheticBusinessEnv / MCP）
- `arlab/reward/*` —— 判分

adapter 的职责只是把 `run_task` 的输出翻译成 verl 的样本结构。

### 路径 B：verl 只有单轮

把多轮折叠进一次 rollout：wrapper 内部跑 N 步工具调用，最终把**完整对话**当作一条 response 交回给 verl，奖励由 `compute_score` 从最终答案算。

代价：PPO 的 importance ratio 只在"整条轨迹"层面成立，不是逐 step 的。对 GRPO 可以接受（GRPO 本来就是轨迹级优势），对 per-token credit assignment 的算法不合适。

---

## 4. 为什么保留自研训练步

不是情怀，是两条实用理由：

1. **数值对照**：同一批 tokenized 样本，分别喂 verl 和 `arlab/train/grpo.py`，比较 `ratio` / `policy_loss` / `kl`。差一个数量级就说明某一侧有 bug。这是面试里能讲的技术点。
2. **兜底**：verl 装不上、或接口要改框架源码时，训练不至于归零。

---

## 5. 上卡顺序（改回 verl 后）

```
1. first_session.sh                        # 体检
2. 起 vLLM + run_baseline --policy llm     # 第一个真实数字（不用 verl）
3. filter_tasks                            # 筛题
4. export_verl                             # 导出数据集
5. pip install verl + 版本探测（第 2 节）   # 决定路径 A/B
6. 跑 verl 自带 example（20 步）            # 证明环境对
7. 接我们的数据集 + 奖励函数                # 出第一条曲线
8. 只在 7 通了之后，再碰多轮                # 最大的不确定性放最后
```

**第 6 步不过，不要碰第 7 步。** 这一条别省。

---

## 6. 止损线

| 现象 | 处理 |
|---|---|
| verl 装不上（CUDA/torch 版本冲突） | 先用自研训练步跑，verl 当简历关键词单独跑 example |
| 多轮接口要改框架源码 | 同上，别改别人的源码 |
| verl 的 reward 签名不匹配 | 改 `verl_reward.py` 的 wrapper，不动核心逻辑 |
| 1000 步 reward 不动 | 回第 3 步查难度分布，别调学习率 |
