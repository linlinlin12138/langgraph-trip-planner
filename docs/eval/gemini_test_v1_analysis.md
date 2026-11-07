# 配对分析 · gemini_test_v1

由 `python -m trip_agents.evals.analyze evals/results/gemini_test_v1` 从 `records.jsonl` 生成（运行目录未修改）。模型 gemini / `gemini-3.8-flash`（推理强度 low），Judge openai / `gpt-6.1-sol`，数据集 `evals/datasets/test.jsonl`（140 条），每组 1 次，记录 420 条。

## 配对翻转（可行样本）

同一样本、同一轮，只有一方通过硬约束的样本数。

| 对比 (A vs B) | 只有 A 通过 | 只有 B 通过 | 配对样本数 |
|---|---|---|---|
| full vs no_revision | 22 | 0 | 131 |
| full vs baseline_catalog | 46 | 1 | 131 |
| no_revision vs baseline_catalog | 29 | 6 | 131 |

## 合格计划质量分：配对差值与 95% 置信区间

两个系统都合格的样本上 B − A 的平均差值；置信区间为配对差值的 bootstrap（5000 次，seed=0）。

| 对比 (A vs B) | 配对样本数 | Δ 总分 | 95% CI | Δ 丰富度 | Δ 偏好符合度 | Δ 合理性 |
|---|---|---|---|---|---|---|
| full vs no_revision | 101 | -0.13 | [-0.35, +0.09] | -0.08 | -0.03 | -0.32 |
| full vs baseline_catalog | 77 | +2.90 | [+1.60, +4.16] | +4.04 | +3.25 | +1.29 |
| no_revision vs baseline_catalog | 72 | +2.89 | [+1.50, +4.26] | +4.35 | +3.00 | +1.28 |

## 合格计划的平均景点数

“全部”是各自所有合格计划的平均；“配对”只算两个系统都合格的样本（排除了“一方只在容易样本上合格”的影响）。

| 对比 (A vs B) | A 全部 | B 全部 | A 配对 | B 配对 | 配对样本数 |
|---|---|---|---|---|---|
| full vs no_revision | 8.59 | 7.60 | 7.59 | 7.60 | 101 |
| full vs baseline_catalog | 8.59 | 8.26 | 7.35 | 8.21 | 77 |
| no_revision vs baseline_catalog | 7.60 | 8.26 | 7.10 | 8.03 | 72 |

## 失败归因（可行样本）

误判不可行：系统判定预算不足（`flagged_infeasible`）但标准答案可行，没有尝试回退；回退耗尽：回退次数达到 2（含最后一轮纯规则修复）仍不通过；其他：其余情况。

| 系统 | 失败数 | 误判不可行 | 回退耗尽 | 其他 |
|---|---|---|---|---|
| full | 8 | 5 | 3 | 0 |
| no_revision | 30 | 5 | 0 | 25 |

