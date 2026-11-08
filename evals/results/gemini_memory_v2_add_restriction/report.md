# 记忆消融评测

| 项 | 值 |
|---|---|
| 日期 | 2026-11-08T01:10:39 |
| git commit | `6caa937` |
| 模型 | gemini / `gemini-3.8-flash`（推理强度 low） |
| 场景数 | 10（只跑 add_restriction；与 --n 40 全集同 id、同 seed） |
| 条件 | 只跑有记忆 |
| seed | 7 |
| 每组运行次数 | 1 |

| 场景 | 系统 | 样本数 | 无需追问即可规划 | 第二次会话字段全对 | 硬约束通过（对照第二次会话标准答案） | 平均 token |
|---|---|---|---|---|---|---|
| add_restriction | full（有记忆） | 10 | 100.0% | 100.0% | 100.0% | 25644 |

说明：

- 两次会话的请求文本都是模板生成的（`template_rewrite`），和主评测集一样。
- 估算费用合计 $0.67（两次会话都计入；无记忆条件不跑第一次会话）。
- 自动模式下不能追问用户。no_memory 缺少预算、人数等信息时无法规划，实际使用中它会转为向用户追问（多一轮交互）。

字段级准确率（有记忆）：

- add_restriction：destination 100%，days 100%，travelers 100%，budget_total 100%，dietary_restrictions 100%，pace 100%，hotel_tier 100%

字段级错误数（有记忆，第二次会话抽取结果和标准答案不一致的场景数）：

| 场景 | destination | days | travelers | budget_total | dietary_restrictions | pace | hotel_tier |
|---|---|---|---|---|---|---|---|
| add_restriction（10） | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

记忆错误的两个方向（有记忆）：

- add_restriction（用户这次新增了饮食限制）：**漏掉新限制 0/10**，丢掉记忆里的旧限制 0/10。出错的代价：可能给过敏的人安排不能吃的东西。标准答案是新旧限制的并集（用户没有撤销旧限制）。
