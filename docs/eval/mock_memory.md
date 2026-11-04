# 记忆消融评测

> ⚠️ **MOCK 模式**：mock 模型用规则抽取偏好，这里验证的是记忆读写与优先级逻辑，不代表真实模型的抽取能力。

| 项 | 值 |
|---|---|
| 日期 | 2026-11-04T16:53:21 |
| git commit | `921589a` |
| 模型 | mock / `mock-planner-v1` |
| 场景数 | 30（三类各 1/3） |
| seed | 7 |
| 每组运行次数 | 1 |

| 场景 | 系统 | 样本数 | 无需追问即可规划 | 第二次会话字段全对 | 硬约束通过（对照第二次会话标准答案） | 平均 token |
|---|---|---|---|---|---|---|
| inherit | full（有记忆） | 10 | 100.0% | 100.0% | 100.0% | 9549 |
| inherit | no_memory | 10 | 0.0% | 0.0% | 0.0% | 344 |
| override | full（有记忆） | 10 | 100.0% | 100.0% | 100.0% | 11459 |
| override | no_memory | 10 | 0.0% | 0.0% | 0.0% | 372 |
| past_trip | full（有记忆） | 10 | 100.0% | 100.0% | 100.0% | 10008 |
| past_trip | no_memory | 10 | 0.0% | 0.0% | 0.0% | 361 |

说明：

- 自动模式下不能追问用户。no_memory 缺少预算、人数等信息时无法规划，实际使用中它会转为向用户追问（多一轮交互）。
- mock 的偏好写入和读取用的是同一套规则，所以 full 的准确率由构造决定，接入真实模型后才能衡量抽取质量。

字段级准确率（有记忆）：

- inherit：destination 100%，days 100%，travelers 100%，budget_total 100%，dietary_restrictions 100%，pace 100%，hotel_tier 100%
- override：destination 100%，days 100%，travelers 100%，budget_total 100%，dietary_restrictions 100%，pace 100%，hotel_tier 100%
- past_trip：destination 100%，days 100%，travelers 100%，budget_total 100%，dietary_restrictions 100%，pace 100%，hotel_tier 100%
