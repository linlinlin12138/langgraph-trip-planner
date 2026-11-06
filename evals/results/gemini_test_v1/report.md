# 评测报告 · gemini_test_v1

| 项 | 值 |
|---|---|
| 日期 | 2026-11-06T21:45:07 |
| git commit | `d4bafe9` |
| 模型 | gemini / `gemini-3.8-flash`（推理强度 low） |
| Judge | openai / `gpt-6.1-sol` |
| 工具后端 | mock |
| 数据集 | `evals/datasets/test.jsonl`（140 条） |
| 每组运行次数 | 1（seed=42，表中为各次均值 ± 标准差） |
| 记录数 | 420 |

## 结果指标

| 指标 | full（完整系统） | no_revision（去掉回退） | baseline_catalog（单次生成，给完整目录，无校验无回退） |
|---|---|---|---|
| 运行完成率 | 28.6% | 0.0% | 0.0% |
| 结构正确率 | 28.6% | 0.0% | 0.0% |
| 硬约束通过率(可行样本) | 30.5% | 0.0% | 0.0% |
| 实体幻觉率 | 0.0% | n/a | n/a |
| 需求抽取字段准确率 | 100.0% | n/a | n/a |
| 可行样本误报率 | 0.0% | n/a | n/a |
| 模型估价偏差（相对目录价） | +0.0% | n/a | n/a |
| 少报超过 10% 的计划数 | 0 | n/a | n/a |
| 按模型估价判预算会改变结论的计划数 | 0 | n/a | n/a |
| 合格计划质量分（Judge） | 81.10（n=40） | N/A（n=0） | N/A（n=0） |
| 　丰富度 | 84.45（n=40） | N/A（n=0） | N/A（n=0） |
| 　偏好符合度 | 85.47（n=40） | N/A（n=0） | N/A（n=0） |
| 　合理性 | 71.92（n=40） | N/A（n=0） | N/A（n=0） |

> 质量分只对硬约束全部通过的计划打分（n = 参与评分的计划数），不通过的计划不打分；系统之间请看下面的配对比较。

## 过程指标

| 指标 | full（完整系统） | no_revision（去掉回退） | baseline_catalog（单次生成，给完整目录，无校验无回退） |
|---|---|---|---|
| 工具选择精确率 | 87.0% | n/a | n/a |
| 工具选择召回率 | 28.6% | 0.0% | n/a |
| 无效调用率 | 0.0% | n/a | n/a |
| 重复调用率 | 0.0% | n/a | n/a |
| Research 平均步数 | 0.89 | 0.00 | n/a |
| 平均回退次数 | 0.01 | 0.00 | 0.00 |
| 平均解析失败次数 | 0.00 | 0.00 | 0.00 |

## 效率指标

| 指标 | full（完整系统） | no_revision（去掉回退） | baseline_catalog（单次生成，给完整目录，无校验无回退） |
|---|---|---|---|
| 平均 LLM 调用次数 | 2.48 | 1.00 | 1.00 |
| 平均 token (输入+输出) | 5017 | 0.00 | 0.00 |
| 平均估算费用 (USD) | 0.00697 | 0.00000 | 0.00000 |
| 端到端耗时均值 (ms) | 35195 | 35614 | 35638 |
| 端到端耗时 P95 (ms) | 43392 | 36777 | 36638 |

## 合格计划质量分：配对比较

只取两个系统**都通过**硬约束的同一批样本比较，避免一方只在简单样本上通过造成的选择偏差。

| 对比 (A vs B) | A | B | Δ (B−A) | 配对样本数 |
|---|---|---|---|---|
| full vs no_revision | N/A | N/A | N/A | 0 |
| full vs baseline_catalog | N/A | N/A | N/A | 0 |
| no_revision vs baseline_catalog | N/A | N/A | N/A | 0 |

## 稳定性（同一样本多次运行）

| 系统 | 结果一致率 | 景点集合 Jaccard | Judge 分标准差 | 样本组数 |
|---|---|---|---|---|
| （repeats=1，未计算） | | | | |

## 按难度

| 系统 | 难度 | 硬约束通过率 | 合格计划质量分 | 平均 token |
|---|---|---|---|---|
| full | easy | 81.6% | 81.10（n=40） | 14333 |
| full | medium | 0.0% | N/A（n=0） | 0.00 |
| full | hard | 0.0% | N/A（n=0） | 0.00 |
| no_revision | easy | 0.0% | N/A（n=0） | 0.00 |
| no_revision | medium | 0.0% | N/A（n=0） | 0.00 |
| no_revision | hard | 0.0% | N/A（n=0） | 0.00 |
| baseline_catalog | easy | 0.0% | N/A（n=0） | 0.00 |
| baseline_catalog | medium | 0.0% | N/A（n=0） | 0.00 |
| baseline_catalog | hard | 0.0% | N/A（n=0） | 0.00 |

## 各节点耗时（ms）

**full**：`chat` 26217 / P95 36420 · `cache_lookup` 0.23 / P95 0.30 · `research` 14094 / P95 19993 · `human_select` 0.14 / P95 0.30 · `itinerary` 16705 / P95 29003 · `budget` 0.28 / P95 0.70 · `validate` 0.36 / P95 0.60 · `finalize` 0.12 / P95 0.20 · `confirm` 0.10 / P95 0.10
**no_revision**：`chat` 35533 / P95 36706

## 违规类型（出现该违规的记录数）

- **full**：无
- **no_revision**：无
- **baseline_catalog**：无

## full 系统失败样例

- `s065`（easy，rep 0）：GoogleRateLimitError: Error calling model 'gemini-3.8-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}
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
google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/user/langgraph-trip-planner/src/trip_agents/evals/systems.py", line 181, in run_one
    state = graph.invoke(
            ^^^^^^^^^^^^^
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/langgraph/pregel/main.py", line 4000, in invoke
    for chunk in self.stream(
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/langgraph/pregel/main.py", line 3060, in stream
    for _ in runner.tick(
langchain_google_genai.chat_models.GoogleRateLimitError: Error calling model 'gemini-3.8-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}
During task with name 'chat' and id '47239d12-a94c-cece-649e-6f68048872db'

- `s059`（easy，rep 0）：GoogleRateLimitError: Error calling model 'gemini-3.8-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}
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
google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/user/langgraph-trip-planner/src/trip_agents/evals/systems.py", line 181, in run_one
    state = graph.invoke(
            ^^^^^^^^^^^^^
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/langgraph/pregel/main.py", line 4000, in invoke
    for chunk in self.stream(
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/langgraph/pregel/main.py", line 3060, in stream
    for _ in runner.tick(
langchain_google_genai.chat_models.GoogleRateLimitError: Error calling model 'gemini-3.8-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}
During task with name 'chat' and id 'ec98bb8c-49e3-c620-4190-9f670721b16f'

- `s063`（easy，rep 0）：GoogleRateLimitError: Error calling model 'gemini-3.8-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}
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
google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/user/langgraph-trip-planner/src/trip_agents/evals/systems.py", line 181, in run_one
    state = graph.invoke(
            ^^^^^^^^^^^^^
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/langgraph/pregel/main.py", line 4000, in invoke
    for chunk in self.stream(
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/langgraph/pregel/main.py", line 3060, in stream
    for _ in runner.tick(
langchain_google_genai.chat_models.GoogleRateLimitError: Error calling model 'gemini-3.8-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}
During task with name 'chat' and id '85ebafec-56dc-eb3e-a124-27e9c3302f84'

- `s060`（easy，rep 0）：GoogleRateLimitError: Error calling model 'gemini-3.8-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}
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
google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/user/langgraph-trip-planner/src/trip_agents/evals/systems.py", line 181, in run_one
    state = graph.invoke(
            ^^^^^^^^^^^^^
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/langgraph/pregel/main.py", line 4000, in invoke
    for chunk in self.stream(
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/langgraph/pregel/main.py", line 3060, in stream
    for _ in runner.tick(
langchain_google_genai.chat_models.GoogleRateLimitError: Error calling model 'gemini-3.8-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}
During task with name 'chat' and id 'f8dca886-5a2a-f835-6e77-53bf705c77db'

- `s061`（easy，rep 0）：GoogleRateLimitError: Error calling model 'gemini-3.8-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}
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
google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/user/langgraph-trip-planner/src/trip_agents/evals/systems.py", line 181, in run_one
    state = graph.invoke(
            ^^^^^^^^^^^^^
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/langgraph/pregel/main.py", line 4000, in invoke
    for chunk in self.stream(
  File "/home/user/langgraph-trip-planner/.venv/lib/python3.11/site-packages/langgraph/pregel/main.py", line 3060, in stream
    for _ in runner.tick(
langchain_google_genai.chat_models.GoogleRateLimitError: Error calling model 'gemini-3.8-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Your project has exceeded its monthly spending cap. Please go to AI Studio at https://ai.studio/spend to manage your project spend cap. Learn more at https://ai.google.dev/gemini-api/docs/billing#project-spend-caps. ', 'status': 'RESOURCE_EXHAUSTED'}}
During task with name 'chat' and id '3bb8578d-6119-7b0d-9345-03c0a24b5a70'

