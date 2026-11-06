# 评测报告 · gemini_dev_v3

| 项 | 值 |
|---|---|
| 日期 | 2026-11-06T02:58:42 |
| git commit | `ace6937` |
| 模型 | gemini / `gemini-3.8-flash` |
| Judge | openai / `gpt-6.1-sol` |
| 工具后端 | mock |
| 数据集 | `evals/datasets/dev.jsonl`（5 条） |
| 每组运行次数 | 1（seed=42，表中为各次均值 ± 标准差） |
| 记录数 | 15 |

## 结果指标

| 指标 | full（完整系统） | no_revision（去掉回退） | baseline（单次生成，无检索无校验） |
|---|---|---|---|
| 运行完成率 | 80.0% | 100.0% | 100.0% |
| 结构正确率 | 80.0% | 100.0% | 100.0% |
| 硬约束通过率(可行样本) | 80.0% | 80.0% | 0.0% |
| 实体幻觉率 | 0.0% | 0.0% | 54.9% |
| 需求抽取字段准确率 | 100.0% | 100.0% | n/a |
| 可行样本误报率 | 0.0% | 0.0% | n/a |
| 模型估价偏差（相对目录价） | +0.0% | +0.0% | -10.3% |
| 少报超过 10% 的计划数 | 0 | 0 | 2 |
| 按模型估价判预算会改变结论的计划数 | 0 | 0 | 1 |
| 合格计划质量分（Judge） | 80.43（n=4） | 80.12（n=4） | N/A（n=0） |
| 　丰富度 | 85.75（n=4） | 86.75（n=4） | N/A（n=0） |
| 　偏好符合度 | 84.25（n=4） | 85.00（n=4） | N/A（n=0） |
| 　合理性 | 70.00（n=4） | 67.00（n=4） | N/A（n=0） |

> 质量分只对硬约束全部通过的计划打分（n = 参与评分的计划数），不通过的计划不打分；系统之间请看下面的配对比较。

## 过程指标

| 指标 | full（完整系统） | no_revision（去掉回退） | baseline（单次生成，无检索无校验） |
|---|---|---|---|
| 工具选择精确率 | 100.0% | 100.0% | n/a |
| 工具选择召回率 | 100.0% | 100.0% | n/a |
| 无效调用率 | 2.8% | 0.0% | n/a |
| 重复调用率 | 0.0% | 0.0% | n/a |
| Research 平均步数 | 3.20 | 3.00 | n/a |
| 平均回退次数 | 0.00 | 0.00 | 0.00 |
| 平均解析失败次数 | 0.00 | 0.00 | 0.00 |

## 效率指标

| 指标 | full（完整系统） | no_revision（去掉回退） | baseline（单次生成，无检索无校验） |
|---|---|---|---|
| 平均 LLM 调用次数 | 8.60 | 7.40 | 1.00 |
| 平均 token (输入+输出) | 41580 | 34148 | 2757 |
| 平均估算费用 (USD) | 0.08531 | 0.06989 | 0.00972 |
| 端到端耗时均值 (ms) | 348498 | 104190 | 20691 |
| 端到端耗时 P95 (ms) | 1406944 | 192214 | 28317 |

## 合格计划质量分：配对比较

只取两个系统**都通过**硬约束的同一批样本比较，避免一方只在简单样本上通过造成的选择偏差。

| 对比 (A vs B) | A | B | Δ (B−A) | 配对样本数 |
|---|---|---|---|---|
| full vs no_revision | 80.42 | 80.12 | -0.30 | 4 |
| full vs baseline | N/A | N/A | N/A | 0 |
| no_revision vs baseline | N/A | N/A | N/A | 0 |

## 稳定性（同一样本多次运行）

| 系统 | 结果一致率 | 景点集合 Jaccard | Judge 分标准差 | 样本组数 |
|---|---|---|---|---|
| （repeats=1，未计算） | | | | |

## 按难度

| 系统 | 难度 | 硬约束通过率 | 合格计划质量分 | 平均 token |
|---|---|---|---|---|
| full | easy | 100.0% | 80.33（n=3） | 28481 |
| full | medium | 100.0% | 80.70（n=1） | 29266 |
| full | hard | 0.0% | N/A（n=0） | 93194 |
| no_revision | easy | 100.0% | 79.60（n=3） | 27888 |
| no_revision | medium | 100.0% | 81.70（n=1） | 29245 |
| no_revision | hard | 0.0% | N/A（n=0） | 57829 |
| baseline | easy | 0.0% | N/A（n=0） | 2433 |
| baseline | medium | 0.0% | N/A（n=0） | 2639 |
| baseline | hard | 0.0% | N/A（n=0） | 3848 |

## 各节点耗时（ms）

**full**：`chat` 4742 / P95 5328 · `cache_lookup` 0.22 / P95 0.30 · `research` 18371 / P95 24645 · `human_select` 0.14 / P95 0.30 · `itinerary` 322729 / P95 1366911 · `budget` 2024 / P95 10118 · `validate` 0.44 / P95 0.50 · `finalize` 0.10 / P95 0.10 · `confirm` 0.10 / P95 0.10
**no_revision**：`chat` 4306 / P95 4950 · `cache_lookup` 0.22 / P95 0.40 · `research` 18381 / P95 20681 · `human_select` 0.12 / P95 0.20 · `itinerary` 79281 / P95 157208 · `budget` 2035 / P95 10175 · `validate` 0.38 / P95 0.50 · `finalize` 0.10 / P95 0.10 · `confirm` 0.10 / P95 0.10

## 违规类型（出现该违规的记录数）

- **full**：无
- **no_revision**：`over_budget` 1
- **baseline**：`unknown_ref` 5、`closed` 4、`unknown_hotel` 2、`time_overlap` 1、`over_budget` 1

## full 系统失败样例

- `s155`（hard，rep 0）：GoogleAPIError: 504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'message': 'Deadline expired before operation could complete.', 'status': 'DEADLINE_EXCEEDED'}}
Traceback (most recent call last):
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/langchain_google_genai/chat_models.py", line 4188, in _generate
    response: GenerateContentResponse = self.client.models.generate_content(
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/google/genai/models.py", line 6370, in generate_content
    response = self._generate_content_with_continuation(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/google/genai/models.py", line 6211, in _generate_content_with_continuation
    response = self._generate_content(
               ^^^^^^^^^^^^^^^^^^^^^^^
google.genai.errors.ServerError: 504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'message': 'Deadline expired before operation could complete.', 'status': 'DEADLINE_EXCEEDED'}}

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/user/langgraph-trip-planner/src/trip_agents/evals/systems.py", line 137, in run_one
    state = graph.invoke(
            ^^^^^^^^^^^^^
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/langgraph/pregel/main.py", line 4000, in invoke
    for chunk in self.stream(
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/langgraph/pregel/main.py", line 3060, in stream
    for _ in runner.tick(
langchain_google_genai.chat_models.GoogleAPIError: 504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'message': 'Deadline expired before operation could complete.', 'status': 'DEADLINE_EXCEEDED'}}
During task with name 'itinerary' and id '91ecbb55-d463-263c-6570-a287405ef6a9'

