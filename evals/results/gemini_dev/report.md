# 评测报告 · gemini_dev

| 项 | 值 |
|---|---|
| 日期 | 2026-11-05T19:55:50 |
| git commit | `5356d23` |
| 模型 | gemini / `gemini-3.8-flash` |
| Judge | 未启用 |
| 工具后端 | mock |
| 数据集 | `evals/datasets/dev.jsonl`（5 条） |
| 每组运行次数 | 1（seed=42，表中为各次均值 ± 标准差） |
| 记录数 | 20 |

## 结果指标

| 指标 | full（完整系统） | no_revision（去掉回退） | no_research（去掉检索） | baseline（单次生成，无检索无校验） |
|---|---|---|---|---|
| 运行完成率 | 100.0% | 100.0% | 100.0% | 100.0% |
| 结构正确率 | 100.0% | 100.0% | 100.0% | 100.0% |
| 硬约束通过率(可行样本) | 100.0% | 100.0% | 0.0% | 0.0% |
| 实体幻觉率 | 0.0% | 0.0% | 67.0% | 69.9% |
| 需求抽取字段准确率 | 100.0% | 100.0% | 100.0% | n/a |
| 可行样本误报率 | 0.0% | 0.0% | 0.0% | n/a |

## 过程指标

| 指标 | full（完整系统） | no_revision（去掉回退） | no_research（去掉检索） | baseline（单次生成，无检索无校验） |
|---|---|---|---|---|
| 工具选择精确率 | 100.0% | 100.0% | n/a | n/a |
| 工具选择召回率 | 100.0% | 100.0% | n/a | n/a |
| 无效调用率 | 0.0% | 0.0% | n/a | n/a |
| 重复调用率 | 0.0% | 0.0% | n/a | n/a |
| Research 平均步数 | 3.40 | 3.40 | n/a | n/a |
| 平均回退次数 | 0.00 | 0.00 | 0.00 | 0.00 |
| 平均解析失败次数 | 0.20 | 0.20 | 0.00 | 0.00 |

## 效率指标

| 指标 | full（完整系统） | no_revision（去掉回退） | no_research（去掉检索） | baseline（单次生成，无检索无校验） |
|---|---|---|---|---|
| 平均 LLM 调用次数 | 7.80 | 7.80 | 4.20 | 1.00 |
| 平均 token (输入+输出) | 31953 | 31922 | 13769 | 2466 |
| 平均估算费用 (USD) | 0.06934 | 0.06933 | 0.04460 | 0.00863 |
| 端到端耗时均值 (ms) | 120784 | 121389 | 93230 | 19763 |
| 端到端耗时 P95 (ms) | 197077 | 199552 | 261285 | 30007 |

## 稳定性（同一样本多次运行）

| 系统 | 结果一致率 | 景点集合 Jaccard | Judge 分标准差 | 样本组数 |
|---|---|---|---|---|
| （repeats=1，未计算） | | | | |

## 按难度

| 系统 | 难度 | 硬约束通过率 | Judge 总分 | 平均 token |
|---|---|---|---|---|
| full | easy | 100.0% | n/a | 25950 |
| full | medium | 100.0% | n/a | 28294 |
| full | hard | 100.0% | n/a | 53619 |
| no_revision | easy | 100.0% | n/a | 25950 |
| no_revision | medium | 100.0% | n/a | 28294 |
| no_revision | hard | 100.0% | n/a | 53465 |
| no_research | easy | 0.0% | n/a | 7745 |
| no_research | medium | 0.0% | n/a | 9115 |
| no_research | hard | 0.0% | n/a | 36495 |
| baseline | easy | 0.0% | n/a | 2349 |
| baseline | medium | 0.0% | n/a | 1762 |
| baseline | hard | 0.0% | n/a | 3520 |

## 各节点耗时（ms）

**full**：`chat` 5277 / P95 6582 · `cache_lookup` 0.18 / P95 0.20 · `research` 19363 / P95 24702 · `human_select` 0.12 / P95 0.20 · `itinerary` 94946 / P95 164583 · `budget` 0.16 / P95 0.20 · `validate` 0.36 / P95 0.40 · `finalize` 0.10 / P95 0.10 · `confirm` 0.10 / P95 0.10
**no_revision**：`chat` 5496 / P95 6876 · `cache_lookup` 0.20 / P95 0.30 · `research` 19232 / P95 23828 · `human_select` 0.10 / P95 0.10 · `itinerary` 96364 / P95 168773 · `budget` 0.16 / P95 0.20 · `validate` 0.30 / P95 0.40 · `finalize` 0.10 / P95 0.10 · `confirm` 0.08 / P95 0.10
**no_research**：`chat` 5262 / P95 6194 · `cache_lookup` 0.18 / P95 0.20 · `research` 0.10 / P95 0.10 · `human_select` 0.10 / P95 0.10 · `itinerary` 87890 / P95 255206 · `budget` 0.16 / P95 0.20 · `validate` 0.34 / P95 0.60 · `finalize` 0.14 / P95 0.30 · `confirm` 0.10 / P95 0.10

## 违规类型（出现该违规的记录数）

- **full**：无
- **no_revision**：无
- **no_research**：`unknown_ref` 5、`missing_hotel` 5、`closed` 2
- **baseline**：`missing_day` 5、`extra_day` 5、`unknown_ref` 5、`unknown_hotel` 5、`wrong_date` 5、`closed` 1
