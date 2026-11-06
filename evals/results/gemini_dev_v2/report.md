# 评测报告 · gemini_dev_v2

| 项 | 值 |
|---|---|
| 日期 | 2026-11-05T22:50:53 |
| git commit | `f85270f` |
| 模型 | gemini / `gemini-3.8-flash` |
| Judge | openai / `gpt-6.1-sol` |
| 工具后端 | mock |
| 数据集 | `evals/datasets/dev.jsonl`（5 条） |
| 每组运行次数 | 1（seed=42，表中为各次均值 ± 标准差） |
| 记录数 | 20 |

## 结果指标

| 指标 | full（完整系统） | no_revision（去掉回退） | baseline（单次生成，无检索无校验） | no_research |
|---|---|---|---|---|
| 运行完成率 | 100.0% | 100.0% | 100.0% | 100.0% |
| 结构正确率 | 100.0% | 100.0% | 100.0% | 100.0% |
| 硬约束通过率(可行样本) | 100.0% | 80.0% | 0.0% | 0.0% |
| 实体幻觉率 | 0.0% | 0.0% | 54.9% | 48.4% |
| 需求抽取字段准确率 | 100.0% | 100.0% | n/a | 100.0% |
| 可行样本误报率 | 0.0% | 0.0% | n/a | 0.0% |
| 模型估价偏差（相对目录价） | +0.0% | +0.0% | -10.3% | -18.7% |
| 少报超过 10% 的计划数 | 0 | 0 | 2 | 2 |
| 按模型估价判预算会改变结论的计划数 | 0 | 0 | 1 | 0 |
| 合格计划质量分（Judge） | 79.00（n=5） | 80.33（n=4） | N/A（n=0） | N/A（n=0） |
| 　丰富度 | 84.40（n=5） | 86.25（n=4） | N/A（n=0） | N/A（n=0） |
| 　偏好符合度 | 82.00（n=5） | 84.75（n=4） | N/A（n=0） | N/A（n=0） |
| 　合理性 | 69.60（n=5） | 68.50（n=4） | N/A（n=0） | N/A（n=0） |

> 质量分只对硬约束全部通过的计划打分（n = 参与评分的计划数），不通过的计划不打分；系统之间请看下面的配对比较。

## 过程指标

| 指标 | full（完整系统） | no_revision（去掉回退） | baseline（单次生成，无检索无校验） | no_research |
|---|---|---|---|---|
| 工具选择精确率 | 100.0% | 100.0% | n/a | n/a |
| 工具选择召回率 | 100.0% | 100.0% | n/a | 0.0% |
| 无效调用率 | 2.8% | 0.0% | n/a | n/a |
| 重复调用率 | 0.0% | 0.0% | n/a | n/a |
| Research 平均步数 | 3.20 | 3.00 | n/a | 0.00 |
| 平均回退次数 | 0.40 | 0.00 | 0.00 | 0.00 |
| 平均解析失败次数 | 0.00 | 0.00 | 0.00 | 0.00 |

## 效率指标

| 指标 | full（完整系统） | no_revision（去掉回退） | baseline（单次生成，无检索无校验） | no_research |
|---|---|---|---|---|
| 平均 LLM 调用次数 | 9.80 | 7.40 | 1.00 | 4.20 |
| 平均 token (输入+输出) | 57434 | 33484 | 2757 | 18503 |
| 平均估算费用 (USD) | 0.13134 | 0.06749 | 0.00972 | 0.06204 |
| 端到端耗时均值 (ms) | 306472 | 100779 | 20921 | 156511 |
| 端到端耗时 P95 (ms) | 1195136 | 176825 | 29100 | 416902 |

## 合格计划质量分：配对比较

只取两个系统**都通过**硬约束的同一批样本比较，避免一方只在简单样本上通过造成的选择偏差。

| 对比 (A vs B) | A | B | Δ (B−A) | 配对样本数 |
|---|---|---|---|---|
| full vs no_revision | 79.28 | 80.33 | +1.05 | 4 |
| full vs baseline | N/A | N/A | N/A | 0 |
| full vs no_research | N/A | N/A | N/A | 0 |
| no_revision vs baseline | N/A | N/A | N/A | 0 |
| no_revision vs no_research | N/A | N/A | N/A | 0 |
| baseline vs no_research | N/A | N/A | N/A | 0 |

## 稳定性（同一样本多次运行）

| 系统 | 结果一致率 | 景点集合 Jaccard | Judge 分标准差 | 样本组数 |
|---|---|---|---|---|
| （repeats=1，未计算） | | | | |

## 按难度

| 系统 | 难度 | 硬约束通过率 | 合格计划质量分 | 平均 token |
|---|---|---|---|---|
| full | easy | 100.0% | 78.80（n=3） | 28481 |
| full | medium | 100.0% | 80.70（n=1） | 29266 |
| full | hard | 100.0% | 77.90（n=1） | 172460 |
| no_revision | easy | 100.0% | 79.87（n=3） | 27999 |
| no_revision | medium | 100.0% | 81.70（n=1） | 29245 |
| no_revision | hard | 0.0% | N/A（n=0） | 54179 |
| baseline | easy | 0.0% | N/A（n=0） | 2433 |
| baseline | medium | 0.0% | N/A（n=0） | 2639 |
| baseline | hard | 0.0% | N/A（n=0） | 3848 |
| no_research | easy | 0.0% | N/A（n=0） | 8180 |
| no_research | medium | 0.0% | N/A（n=0） | 34199 |
| no_research | hard | 0.0% | N/A（n=0） | 33776 |

## 各节点耗时（ms）

**full**：`chat` 4800 / P95 5766 · `cache_lookup` 0.18 / P95 0.20 · `research` 18334 / P95 25174 · `human_select` 0.10 / P95 0.10 · `itinerary` 279026 / P95 1146141 · `budget` 3651 / P95 18255 · `validate` 0.60 / P95 1.40 · `finalize` 0.10 / P95 0.10 · `confirm` 0.10 / P95 0.10
**no_revision**：`chat` 4823 / P95 7075 · `cache_lookup` 0.14 / P95 0.20 · `research` 18935 / P95 21553 · `human_select` 0.10 / P95 0.10 · `itinerary` 74618 / P95 139805 · `budget` 2200 / P95 10999 · `validate` 0.38 / P95 0.50 · `finalize` 0.10 / P95 0.10 · `confirm` 0.10 / P95 0.10
**no_research**：`chat` 4987 / P95 5758 · `cache_lookup` 0.16 / P95 0.20 · `research` 0.10 / P95 0.10 · `human_select` 0.10 / P95 0.10 · `itinerary` 151463 / P95 411183 · `budget` 0.12 / P95 0.20 · `validate` 0.30 / P95 0.40 · `finalize` 0.10 / P95 0.10 · `confirm` 0.10 / P95 0.10

## 违规类型（出现该违规的记录数）

- **full**：无
- **no_revision**：`over_budget` 1
- **baseline**：`unknown_ref` 5、`closed` 4、`unknown_hotel` 2、`over_budget` 1、`time_overlap` 1
- **no_research**：`missing_hotel` 5、`unknown_ref` 5、`closed` 1
