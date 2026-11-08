# 记忆消融评测

| 项 | 值 |
|---|---|
| 日期 | 2026-11-08T20:03:43 |
| git commit | `9f80a99` |
| 模型 | gemini / `gemini-3.8-flash`（推理强度 low） |
| 场景数 | 20（只跑 inherit、override；与 --n 40 全集同 id、同 seed） |
| 条件 | 只跑有记忆 |
| seed | 7 |
| 每组运行次数 | 1 |

| 场景 | 系统 | 样本数 | 无需追问即可规划 | 第二次会话字段全对 | 硬约束通过（对照第二次会话标准答案） | 平均 token |
|---|---|---|---|---|---|---|
| inherit | full（有记忆） | 10 | 100.0% | 100.0% | 100.0% | 22275 |
| override | full（有记忆） | 10 | 100.0% | 100.0% | 100.0% | 23239 |

说明：

- 两次会话的请求文本都是模板生成的（`template_rewrite`），和主评测集一样。
- 估算费用合计 $1.25（两次会话都计入；无记忆条件不跑第一次会话）。
- 自动模式下不能追问用户。no_memory 缺少预算、人数等信息时无法规划，实际使用中它会转为向用户追问（多一轮交互）。

字段级准确率（有记忆）：

- inherit：destination 100%，days 100%，travelers 100%，budget_total 100%，dietary_restrictions 100%，pace 100%，hotel_tier 100%
- override：destination 100%，days 100%，travelers 100%，budget_total 100%，dietary_restrictions 100%，pace 100%，hotel_tier 100%

字段级错误数（有记忆，第二次会话抽取结果和标准答案不一致的场景数）：

| 场景 | destination | days | travelers | budget_total | dietary_restrictions | pace | hotel_tier |
|---|---|---|---|---|---|---|---|
| inherit（10） | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| override（10） | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

记忆错误的两个方向（有记忆）：

- override（用户这次放宽了限制、改了节奏）：旧饮食限制残留 0/10，旧节奏残留 0/10。出错的代价：行程过于保守。
