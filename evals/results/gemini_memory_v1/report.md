# 记忆消融评测

| 项 | 值 |
|---|---|
| 日期 | 2026-11-08T00:41:20 |
| git commit | `a1d8b49` |
| 模型 | gemini / `gemini-3.8-flash`（推理强度 low） |
| 场景数 | 40（4 类各 1/4） |
| seed | 7 |
| 每组运行次数 | 1 |

| 场景 | 系统 | 样本数 | 无需追问即可规划 | 第二次会话字段全对 | 硬约束通过（对照第二次会话标准答案） | 平均 token |
|---|---|---|---|---|---|---|
| inherit | full（有记忆） | 10 | 100.0% | 60.0% | 100.0% | 22372 |
| inherit | no_memory | 10 | 0.0% | 0.0% | 0.0% | 453 |
| override | full（有记忆） | 10 | 100.0% | 40.0% | 100.0% | 22184 |
| override | no_memory | 10 | 0.0% | 0.0% | 0.0% | 499 |
| past_trip | full（有记忆） | 10 | 100.0% | 100.0% | 100.0% | 23134 |
| past_trip | no_memory | 10 | 0.0% | 0.0% | 0.0% | 452 |
| add_restriction | full（有记忆） | 10 | 100.0% | 0.0% | 10.0% | 25899 |
| add_restriction | no_memory | 10 | 0.0% | 0.0% | 0.0% | 473 |

说明：

- 两次会话的请求文本都是模板生成的（`template_rewrite`），和主评测集一样。
- 估算费用合计 $2.57（两次会话都计入；无记忆条件不跑第一次会话）。
- 自动模式下不能追问用户。no_memory 缺少预算、人数等信息时无法规划，实际使用中它会转为向用户追问（多一轮交互）。

字段级准确率（有记忆）：

- inherit：destination 100%，days 100%，travelers 60%，budget_total 60%，dietary_restrictions 100%，pace 100%，hotel_tier 100%
- override：destination 100%，days 100%，travelers 40%，budget_total 40%，dietary_restrictions 100%，pace 100%，hotel_tier 100%
- past_trip：destination 100%，days 100%，travelers 100%，budget_total 100%，dietary_restrictions 100%，pace 100%，hotel_tier 100%
- add_restriction：destination 100%，days 100%，travelers 60%，budget_total 60%，dietary_restrictions 0%，pace 100%，hotel_tier 100%

字段级错误数（有记忆，第二次会话抽取结果和标准答案不一致的场景数）：

| 场景 | destination | days | travelers | budget_total | dietary_restrictions | pace | hotel_tier |
|---|---|---|---|---|---|---|---|
| inherit（10） | 0 | 0 | 4 | 4 | 0 | 0 | 0 |
| override（10） | 0 | 0 | 6 | 6 | 0 | 0 | 0 |
| past_trip（10） | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| add_restriction（10） | 0 | 0 | 4 | 4 | 10 | 0 | 0 |

记忆错误的两个方向（有记忆）：

- override（用户这次放宽了限制、改了节奏）：旧饮食限制残留 0/10，旧节奏残留 0/10。出错的代价：行程过于保守。
- add_restriction（用户这次新增了饮食限制）：**漏掉新限制 0/10**，丢掉记忆里的旧限制 10/10。出错的代价：可能给过敏的人安排不能吃的东西。标准答案是新旧限制的并集（用户没有撤销旧限制）。
