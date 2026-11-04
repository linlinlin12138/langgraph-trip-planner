# react_from_scratch：不用框架手写 ReAct

这个目录是面试准备的练习：只用模型 SDK，自己实现 ReAct 循环，再和 LangGraph 版的 Research 子图在同一批任务上比较效果、token 和代码量。

## 已经提供的部分

| 文件 | 内容 |
|---|---|
| `types.py` | 消息格式（OpenAI 风格 dict）、`ModelTurn`、`RawToolCall`、`ParsedCall`、`Observation`、`ToolCallLog`、`ReactConfig`、`ReactResult` |
| `registry.py` | `ToolRegistry`：注册工具、导出 JSON Schema、参数校验、执行。出错时返回以 `ERROR` 开头的 observation，不抛异常。`from_trip_tools()` 注册项目里的 5 个旅行工具 |
| `client.py` | `ChatClient` 协议，以及 `OpenAIClient` / `AnthropicClient` / `GeminiClient`（走 OpenAI 兼容接口）。限流、超时、5xx 统一抛 `TransientModelError`。另有离线用的 `ScriptedClient` 和 `MockResearchClient`（与 LangGraph mock 用同一套检索策略） |
| `compare.py` | 对比脚本 |
| `tests/test_loop.py` | 14 条验收测试 |

## 需要你实现的部分：`loop.py`

- `parse_tool_calls(turn)`：把模型输出的原始参数字符串解析成 dict，格式不合法时给出可读的错误。
- `run_react(client, registry, task, *, system_prompt, config)`：完整的循环。契约写在 `loop.py` 顶部的 docstring 里：
  1. 调用模型；遇到 `TransientModelError` 重试，次数有上限，并累计 usage；
  2. 没有 tool call 时结束，`stopped_reason="final_answer"`；
  3. 对每个 tool call 依次执行：解析参数 → **重复调用检测**（同名、同参数、之前成功过的调用不再执行，直接返回上次结果并标记）→ **每类工具的轮数上限** → `registry.execute`（参数校验失败也作为 observation 返回给模型）；
  4. 每个 `tool_call_id` 恰好对应一条 `role=tool` 消息，并记一条 `ToolCallLog`；
  5. 达到 `max_steps` 时结束，`stopped_reason="max_steps"`。

## 运行

```bash
make test-react        # 未实现时测试显示 skipped；实现后自动开始运行
```

```bash
make compare-react     # mock 模式，dev 集抽 20 条
```

```bash
uv run python -m react_from_scratch.compare --provider openai --model <模型名> --limit 20   # 真实模型，会产生费用
```

对比的指标包括：检索成功率、工具精确率和召回率（与样本标注的期望工具比较）、步数、无效调用率、重复调用率、token、平均耗时和 P95，以及循环部分的逻辑行数。

## 面试时可以讲的点

- 解析失败、参数非法、未知工具都应该**作为 observation 返回给模型**，而不是抛异常。这正是 ReAct 能自我纠错的原因。
- 重复调用的判定要先把参数规范化（排序后的 JSON），并且只缓存**成功**的调用。失败的调用允许模型修正后重试。
- 轮数上限按工具类别计算，不按步数计算：并行调用时一步里会有多个 call。
- 模型调用的重试只针对可恢复的错误（429、超时、5xx），鉴权错误这类要直接失败。
- 和 LangGraph 版对比：框架提供的是状态持久化、中断恢复、可视化和回调，循环逻辑本身并不复杂。
