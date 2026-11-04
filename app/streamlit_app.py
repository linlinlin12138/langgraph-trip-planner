"""Streamlit demo.  Run:  make demo   (mock mode, no API key needed)

Left: conversation.  Middle: human-in-the-loop candidate picking + final plan.
Right: live trace (nodes, LLM calls / tokens / latency, tool calls, re-plans)
and the long-term memory panel.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from trip_agents.config import get_settings
from trip_agents.graph.deps import GraphOptions
from trip_agents.memory.store import MemoryManager, sqlite_store
from trip_agents.observability.tracer import TraceRecorder
from trip_agents.runtime import PlannerSession, create_planner

st.set_page_config(page_title="Trip Agents", page_icon="🧭", layout="wide")
S = st.session_state
NODE_LABELS = {
    "chat": "Chat：理解需求",
    "cache_lookup": "语义缓存查询",
    "clarify": "等待用户补充",
    "research": "Research：ReAct 检索",
    "human_select": "等待用户挑选候选",
    "itinerary": "Itinerary：逐日规划",
    "budget": "Budget：核算预算",
    "validate": "Validate：规则校验",
    "finalize": "Finalize：汇总",
    "confirm": "等待用户确认",
}
LABEL = {"budget": "经济型", "midrange": "舒适型", "luxury": "高档", "vegetarian": "素食", "halal": "清真", "no_pork": "无猪肉",
         "no_spicy": "不辣", "no_seafood": "无海鲜", "history": "历史", "nature": "自然", "museum": "博物馆", "food": "美食",
         "shopping": "购物", "nightlife": "夜景", "culture": "文化",
         "sunny": "☀️ 晴", "cloudy": "⛅ 多云", "rain": "🌧️ 雨", "snow": "❄️ 雪"}  # fmt: skip
EXAMPLES = [
    "我和女朋友11月20日去杭州玩3天，预算6000，不吃辣，喜欢自然和美食，一定要去灵隐寺",
    "一家三口12月5日去成都玩4天，人均预算2000元，节奏轻松点，想看熊猫",
    "想去西安玩，喜欢历史",  # incomplete -> clarification
]
FAULTS = {
    "itinerary_overlap_once": "首版行程注入时间冲突（演示回退重排）",
    "bad_tool_args_once": "首次工具调用参数错误（演示错误自纠）",
    "bad_json_once": "首次结构化输出 JSON 损坏（演示解析重试）",
}


@st.cache_resource
def memory_store() -> Any:
    return sqlite_store(get_settings().store_db)


def new_session(faults: set[str], user_id: str, cache: bool) -> None:
    graph = create_planner(faults=faults, options=GraphOptions(), store=memory_store(), semantic_cache=cache)
    S.cache_on = cache
    S.tracer = TraceRecorder()  # callback-based: nodes don't know it exists
    S.session = PlannerSession(graph, user_id=user_id, callbacks=[S.tracer])
    S.chat = []
    S.node_log = []
    S.interrupt = None


def run(fn: Any, *args: Any) -> None:
    with st.status("Agent 运行中…", expanded=True) as status:
        for node, _update in fn(*args):
            S.node_log.append(node)
            status.write(f"✅ {NODE_LABELS.get(node, node)}")
        status.update(label="完成", state="complete", expanded=False)
    S.interrupt = S.session.pending_interrupt()
    values = S.session.values
    S.chat = [(m.type, m.content) for m in values.get("messages", []) if m.type in ("human", "ai")]


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("设置")
    cfg = get_settings()
    st.caption(f"模型：`{cfg.llm_provider}` · 工具后端：`{cfg.tool_backend}`")
    if cfg.llm_provider == "mock":
        st.info("Mock 模式：规则模拟的模型，不调用任何外部 API；结果仅用于演示流程。")
    user_id = st.text_input("用户 ID（长期记忆按用户隔离）", value="demo-user")
    cache_on = st.toggle("语义缓存（命中时跳过检索，复用并重新校验已确认行程）", value=cfg.semantic_cache)
    st.markdown("**故障注入**（mock 下演示容错路径）")
    chosen = {k for k, label in FAULTS.items() if st.checkbox(label, key=f"fault_{k}")}
    changed = "session" not in S or S.session.user_id != user_id or S.get("cache_on") != cache_on
    if st.button("开始新规划", type="primary") or changed:
        new_session(chosen, user_id, cache_on)
    if st.button("清空该用户的长期记忆"):
        MemoryManager(memory_store()).clear(user_id)
        st.toast(f"已清空 {user_id} 的画像和历史行程")
    st.caption(f"thread_id: `{S.session.thread_id}`")

left, mid, right = st.columns([1.1, 1.5, 1.0], gap="medium")

# ------------------------------------------------------------------ left: chat
with left:
    st.subheader("💬 对话")
    for role, content in S.chat:
        st.chat_message("user" if role == "human" else "assistant").write(content)
    if not S.chat:
        st.caption("试试这些请求：")
        for i, ex in enumerate(EXAMPLES):
            if st.button(ex, key=f"ex{i}", use_container_width=True):
                S.queued = ex
                st.rerun()
    intr_now = S.interrupt
    asks_defaults = bool(intr_now and intr_now["type"] == "clarify" and intr_now.get("memory_defaults") and not intr_now.get("missing"))
    if asks_defaults and st.button("✅ 沿用这些偏好", type="primary"):
        S.queued = "可以"
        st.rerun()
    placeholder = "描述你的旅行需求，或回答上面的问题"
    text = st.chat_input(placeholder) or S.pop("queued", None)
    if text:
        intr = S.interrupt
        if intr and intr["type"] == "clarify":
            run(S.session.stream, Command(resume=text))
        elif intr is None:
            run(S.session.stream, {"messages": [HumanMessage(content=text)], "user_id": S.session.user_id})
        else:
            st.warning("请先在中间栏完成当前步骤。")
        st.rerun()

# ------------------------------------------------------------------ middle: HITL + plan
with mid:
    intr = S.interrupt
    values = S.session.values
    if intr and intr["type"] == "select_candidates":
        st.subheader("🧑‍⚖️ 请挑选候选（Human-in-the-loop）")
        d = intr["defaults"]
        with st.form("select"):
            hotels = {
                f"{h['name']} · {LABEL[h['tier']]} · ¥{h['price_per_night']:.0f}/晚 · ★{h['rating']}": h["id"] for h in intr["hotels"]
            }
            hotel_default = next((i for i, v in enumerate(hotels.values()) if v == d["hotel_id"]), 0)
            hotel = st.radio("酒店", list(hotels), index=hotel_default) if hotels else None
            attrs = {
                f"{a['name']} [{LABEL[a['category']]}] ¥{a['ticket_price']:.0f} · {a['open_time'][:5]}-{a['close_time'][:5]}": a["id"]
                for a in intr["attractions"]
            }
            attr_sel = st.multiselect("景点", list(attrs), default=[k for k, v in attrs.items() if v in d["attraction_ids"]])
            rests = {
                f"{r['name']} · 人均¥{r['avg_price_per_person']:.0f} · {'/'.join(LABEL[t] for t in r['dietary_tags']) or '-'}": r["id"]
                for r in intr["restaurants"]
            }
            rest_sel = st.multiselect("餐厅", list(rests), default=[k for k, v in rests.items() if v in d["restaurant_ids"]])
            if intr.get("weather"):
                st.caption(
                    "天气："
                    + " · ".join(
                        f"{w['date'][5:]} {LABEL[w['condition']]} {w['temp_low']:.0f}~{w['temp_high']:.0f}℃" for w in intr["weather"]
                    )
                )
            if st.form_submit_button("确认选择，生成行程", type="primary"):
                payload = {
                    "hotel_id": hotels[hotel] if hotel else None,
                    "attraction_ids": [attrs[k] for k in attr_sel],
                    "restaurant_ids": [rests[k] for k in rest_sel],
                }
                run(S.session.stream, Command(resume=payload))
                st.rerun()

    final = values.get("final")
    if final:
        st.subheader("🗺️ 行程")
        badge = "✅ 通过规则校验" if final.passed_validation else "⚠️ 存在未解决问题"
        st.caption(f"{badge} · 修订 {final.revisions} 次 · 预计 ¥{final.budget.total:.0f} / 预算 ¥{final.budget.budget:.0f}")
        if hit := values.get("cache_hit"):
            st.info(f"♻️ 语义缓存命中（相似度 {hit['similarity']:.2f}）：跳过了检索和候选挑选，复用的行程已重新核算并通过校验。")
        for msg in final.remaining_issues:
            if msg.severity == "error":
                st.warning(msg.message)
        weather = {w.date: w for w in values["research"].weather} if values.get("research") else {}
        for day in final.itinerary.days:
            w = weather.get(day.date)
            wtxt = f" · {LABEL[w.condition]} {w.temp_low:.0f}~{w.temp_high:.0f}℃" if w else ""
            title = f"第 {day.day_index + 1} 天 · {day.date}{wtxt} · ¥{day.cost:.0f} · {day.theme}"
            with st.expander(title, expanded=day.day_index == 0):
                for a in day.activities:
                    st.markdown(f"`{a.start_time:%H:%M}-{a.end_time:%H:%M}` {a.title} · ¥{a.est_cost:.0f}")
        st.dataframe(pd.DataFrame([li.model_dump() for li in final.budget.lines]), hide_index=True, use_container_width=True)
        for s in final.budget.suggestions:
            st.markdown(f"- 💡 {s}")
        if intr and intr["type"] == "confirm":
            if st.button("确认行程", type="primary"):
                run(S.session.stream, Command(resume={"confirmed": True}))
                st.rerun()
        elif values.get("status") == "confirmed":
            st.success(
                "行程已确认，已更新长期记忆。" + ("（已写入语义缓存）" if "已写入语义缓存" in values.get("memory_updates", []) else "")
            )
    elif not intr:
        st.info("在左侧输入你的旅行需求开始。")

# ------------------------------------------------------------------ right: trace
with right:
    st.subheader("🔎 执行轨迹")
    if S.node_log:
        st.markdown(" → ".join(f"`{n}`" for n in S.node_log))
        waiting = S.interrupt["type"] if S.interrupt else None
        st.caption(f"当前：{'等待用户（' + waiting + '）' if waiting else NODE_LABELS.get(S.node_log[-1], S.node_log[-1])}")
    tr = S.tracer.trace
    if tr.llm_calls:
        st.markdown(
            f"**LLM 调用** {len(tr.llm_calls)} 次 · **token** 入 {tr.input_tokens} / 出 {tr.output_tokens} · "
            f"**解析重试** {tr.parse_failures} 次"
        )
        per_node: dict[str, dict[str, float]] = {}
        for c in tr.llm_calls:
            d = per_node.setdefault(c.node, {"调用": 0, "token": 0, "LLM 耗时 ms": 0.0})
            d["调用"] += 1
            d["token"] += c.input_tokens + c.output_tokens
            d["LLM 耗时 ms"] += c.latency_ms
        for node, ms in tr.node_latency().items():
            per_node.setdefault(node, {"调用": 0, "token": 0, "LLM 耗时 ms": 0.0})["节点耗时 ms"] = round(sum(ms), 1)
        table = pd.DataFrame(per_node).T.fillna(0)
        table = table[(table["调用"] > 0) | (table.get("节点耗时 ms", 0) >= 1)]
        st.dataframe(table, use_container_width=True)
        if tr.cost_usd is None:
            st.caption("费用：当前模型未配置单价（见 observability/pricing.py）")
        else:
            st.caption(f"估算费用：${tr.cost_usd:.5f} · 模型：{', '.join(sorted({c.model for c in tr.llm_calls}))}")
    calls = values.get("tool_calls", [])
    if calls:
        st.markdown(f"**工具调用 {len(calls)} 次**")
        df = pd.DataFrame(calls)[["step", "name", "ok", "duplicate", "latency_ms", "error"]]
        st.dataframe(df, hide_index=True, use_container_width=True)
    history = values.get("issue_history", [])
    if history:
        st.markdown(f"**规则校验 {len(history)} 轮**")
        for i, issues in enumerate(history):
            errs = [x for x in issues if x.severity == "error"]
            st.markdown(f"- 第 {i + 1} 轮：{'通过' if not errs else f'{len(errs)} 个错误 → 回退重排'}")
            for e in errs[:3]:
                st.caption(f"  · {e.message}")
    for item in values.get("skipped_items", []):
        st.warning(f"跳过 {item.stage}/{item.item}：{item.errors[-1][:80]}")

    st.subheader("🧠 长期记忆")
    mm = MemoryManager(memory_store())
    profile = mm.get_profile(S.session.user_id)
    if profile and profile.describe():
        for line in profile.describe():
            st.markdown(f"- {line}")
    else:
        st.caption("暂无用户画像：确认一次行程后会提取你明确表达过的偏好。")
    if values.get("memory_defaults"):
        st.caption("本次沿用的记忆：" + "；".join(values["memory_defaults"].values()))
    for note in values.get("memory_updates", []):
        st.caption(f"✍️ {note}")
    trips = mm.list_trips(S.session.user_id)
    if trips:
        with st.expander(f"历史行程摘要（{len(trips)} / 最多 {mm.max_trips}）"):
            for t in trips:
                st.markdown(f"- {t.describe()}")
