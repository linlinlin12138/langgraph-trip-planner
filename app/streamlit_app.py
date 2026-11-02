"""Streamlit demo.  Run:  uv run streamlit run app/streamlit_app.py

Left: conversation.  Middle: human-in-the-loop candidate picking + final plan.
Right: live execution panel (nodes, tool calls, re-plans).  Token / latency
tracing and the memory panel arrive in later milestones.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from trip_agents.config import get_settings
from trip_agents.graph.deps import GraphOptions
from trip_agents.runtime import PlannerSession, create_planner

st.set_page_config(page_title="Trip Agents", page_icon="🧭", layout="wide")
S = st.session_state
NODE_LABELS = {
    "chat": "Chat：理解需求",
    "clarify": "等待用户补充",
    "research": "Research：ReAct 检索",
    "human_select": "等待用户挑选候选",
    "itinerary": "Itinerary：逐日规划",
    "budget": "Budget：核算预算",
    "validate": "Validate：规则校验",
    "finalize": "Finalize：汇总",
    "confirm": "等待用户确认",
}
FAULTS = {
    "itinerary_overlap_once": "首版行程注入时间冲突（演示回退重排）",
    "bad_tool_args_once": "首次工具调用参数错误（演示错误自纠）",
    "bad_json_once": "首次结构化输出 JSON 损坏（演示解析重试）",
}


def new_session(faults: set[str]) -> None:
    graph = create_planner(faults=faults, options=GraphOptions())
    S.session = PlannerSession(graph)
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
    chosen = {k for k, label in FAULTS.items() if st.checkbox(label, key=f"fault_{k}")}
    if st.button("开始新规划", type="primary") or "session" not in S:
        new_session(chosen)
    st.caption(f"thread_id: `{S.session.thread_id}`")

left, mid, right = st.columns([1.1, 1.5, 1.0], gap="medium")

# ------------------------------------------------------------------ left: chat
with left:
    st.subheader("💬 对话")
    for role, content in S.chat:
        st.chat_message("user" if role == "human" else "assistant").write(content)
    placeholder = "例：我和女朋友11月20日去杭州玩3天，预算6000，不吃辣，一定要去灵隐寺"
    if text := st.chat_input(placeholder):
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
            hotels = {f"{h['name']} · {h['tier']} · ¥{h['price_per_night']:.0f}/晚 · ★{h['rating']}": h["id"] for h in intr["hotels"]}
            hotel_default = next((i for i, v in enumerate(hotels.values()) if v == d["hotel_id"]), 0)
            hotel = st.radio("酒店", list(hotels), index=hotel_default) if hotels else None
            attrs = {f"{a['name']} [{a['category']}] ¥{a['ticket_price']:.0f}": a["id"] for a in intr["attractions"]}
            attr_sel = st.multiselect("景点", list(attrs), default=[k for k, v in attrs.items() if v in d["attraction_ids"]])
            rests = {
                f"{r['name']} · 人均¥{r['avg_price_per_person']:.0f} · {','.join(r['dietary_tags']) or '-'}": r["id"]
                for r in intr["restaurants"]
            }
            rest_sel = st.multiselect("餐厅", list(rests), default=[k for k, v in rests.items() if v in d["restaurant_ids"]])
            if intr.get("weather"):
                st.caption(
                    "天气："
                    + " · ".join(f"{w['date'][5:]} {w['condition']} {w['temp_low']:.0f}~{w['temp_high']:.0f}℃" for w in intr["weather"])
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
        for day in final.itinerary.days:
            with st.expander(f"第 {day.day_index + 1} 天 · {day.date} · {day.theme}", expanded=day.day_index == 0):
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
            st.success("行程已确认。")
    elif not intr:
        st.info("在左侧输入你的旅行需求开始。")

# ------------------------------------------------------------------ right: trace
with right:
    st.subheader("🔎 执行轨迹")
    if S.node_log:
        st.markdown(" → ".join(f"`{n}`" for n in S.node_log))
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
