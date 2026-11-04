# react_from_scratch：不用框架手写 ReAct

只用模型 SDK 实现的 ReAct 循环，和 LangGraph 版的 Research 子图在同一批任务上比较效果、token 和代码量。

## 文件

| 文件 | 内容 |
|---|---|
| `loop.py` | **ReAct 循环本身**：`parse_tool_calls` 和 `run_react`，契约写在文件顶部的 docstring 里 |
| `types.py` | 消息格式（OpenAI 风格 dict）、`ModelTurn`、`RawToolCall`、`ParsedCall`、`Observation`、`ToolCallLog`、`ReactConfig`、`ReactResult` |
| `registry.py` | `ToolRegistry`：注册工具、导出 JSON Schema、参数校验、执行。出错时返回以 `ERROR` 开头的 observation，不抛异常。`from_trip_tools()` 注册项目里的 5 个旅行工具 |
| `client.py` | `ChatClient` 协议，以及 `OpenAIClient` / `AnthropicClient` / `GeminiClient`（走 OpenAI 兼容接口）。限流、超时、5xx 统一抛 `TransientModelError`。另有离线用的 `ScriptedClient` 和 `MockResearchClient`（与 LangGraph mock 用同一套检索策略） |
| `compare.py` | 对比脚本 |
| `tests/test_loop.py` | 14 条验收测试 |

## 循环做了什么（`loop.py`）

1. **思考**：调用模型。只有 `TransientModelError`（429、超时、5xx）会重试，有次数上限；鉴权、参数错误这类直接失败，因为重试也没用。每次调用都累计 usage。
2. **结束**：模型没有发起工具调用时，`stopped_reason="final_answer"`。
3. **行动和观察**：对每个工具调用依次执行：
   - 解析参数：JSON 格式坏了，或者不是对象，就返回错误 observation；
   - **重复调用检测**：同名、同参数（规范化后比较）、之前**成功过**的调用不再执行，直接返回上次的结果并附上提示；
   - **每类工具的轮数上限**：失败的调用也占一轮；
   - `registry.execute`：参数校验失败、未知工具、工具内部异常，都会作为 observation 返回给模型。
4. 每个 `tool_call_id` 恰好对应一条 `role=tool` 消息，各家 API 都会拒绝缺少结果的工具调用历史。同时为每次调用记一条 `ToolCallLog`。
5. 达到 `max_steps` 时结束，`stopped_reason="max_steps"`。

## 运行

```bash
make test-react
```

```bash
make compare-react     # mock 模式，dev 集抽 20 条
```

```bash
uv run python -m react_from_scratch.compare --provider gemini --model <模型名> --limit 20   # 真实模型，会产生费用
```

## 对比结果（mock）

mock 模式下，dev 集按 seed=42 抽 20 条，每条跑 1 次，完整报告见 [`docs/eval/mock_react_compare.md`](../docs/eval/mock_react_compare.md)。两边用的是**同一套确定性检索策略**，所以检索成功率、工具精确率和召回率、步数完全一致。这说明两种实现的行为等价，但不能说明哪种实现效果更好。

两者真正的差别在代码量和框架提供的能力上：
- 手写循环是 92 个逻辑行，LangGraph 子图是 65 行（不含框架本身）；
- 手写版要自己管理消息历史、重试和状态；
- LangGraph 版额外得到了 checkpoint、中断恢复、子图嵌套和统一的回调。

真实模型下的差异，比如对错误 observation 的自我纠正能力，要等用 Gemini 跑过对比之后才知道。

## 面试时可以讲的点

- 解析失败、参数非法、未知工具都应该**作为 observation 返回给模型**，而不是抛异常。这正是 ReAct 能自我纠错的原因。
- 重复调用的判定要先把参数规范化（排序后的 JSON），并且只缓存**成功**的调用。失败的调用允许模型修正后重试。
- 轮数上限按工具类别计算，不按步数计算：并行调用时，一步里会有多个 call。
- 模型调用的重试只针对可恢复的错误。
- 和 LangGraph 版对比：循环逻辑本身并不复杂，框架的价值在于状态持久化、中断恢复、可视化和回调。
