"""
Phase 15 — Observability: Agent Trace Dashboard Component

Renders the full observability dashboard in the MLForge Streamlit UI.
Displays: Agent Trace, Workflow Progress, Agent Performance, Tool Usage,
          LLM Usage, Decision Timeline, Error Timeline, and Export.

NEVER displays: chain-of-thought, raw prompts, API keys, or secrets.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

# ─── Icon and color mappings ─────────────────────────────────────────────────

_STATUS_ICON = {
    "SUCCESS":  "✅",
    "RUNNING":  "⟳",
    "FAILED":   "❌",
    "WARNING":  "⚠️",
    "SKIPPED":  "⏭️",
    "RETRYING": "🔄",
    "FALLBACK": "↩️",
    None:       "○",
}

_EVENT_COLOR = {
    "WORKFLOW_STARTED":    "#4a9eff",
    "WORKFLOW_COMPLETED":  "#2ecc71",
    "WORKFLOW_FAILED":     "#e74c3c",
    "AGENT_STARTED":       "#3498db",
    "AGENT_COMPLETED":     "#2ecc71",
    "AGENT_FAILED":        "#e74c3c",
    "DECISION":            "#9b59b6",
    "TOOL_STARTED":        "#f39c12",
    "TOOL_COMPLETED":      "#f39c12",
    "TOOL_FAILED":         "#e74c3c",
    "HANDOFF":             "#1abc9c",
    "LLM_REQUEST":         "#95a5a6",
    "LLM_RESPONSE":        "#2ecc71",
    "LLM_RETRY":           "#e67e22",
    "LLM_FALLBACK":        "#e74c3c",
    "TRAINING_STARTED":    "#3498db",
    "TRAINING_COMPLETED":  "#2ecc71",
    "EVALUATION_STARTED":  "#3498db",
    "EVALUATION_COMPLETED":"#2ecc71",
    "REQUIREMENT_PASSED":  "#2ecc71",
    "REQUIREMENT_FAILED":  "#e74c3c",
    "DEBUGGER_ANALYSIS":   "#e67e22",
    "REVIEW_STARTED":      "#9b59b6",
    "REVIEW_COMPLETED":    "#2ecc71",
    "CHECKPOINT_CREATED":  "#1abc9c",
    "CHECKPOINT_RESTORED": "#1abc9c",
    "ERROR":               "#e74c3c",
    "STATE_UPDATED":       "#7f8c8d",
}

_AGENT_WORKFLOW_ORDER = [
    "orchestrator", "data_analyst", "feature_engineer",
    "model_engineer", "training", "evaluation",
    "debugger", "reviewer",
]

_AGENT_LABELS = {
    "orchestrator":    "Orchestrator",
    "data_analyst":    "Data Agent",
    "feature_engineer":"Feature Agent",
    "model_engineer":  "Model Agent",
    "training":        "Training Agent",
    "evaluation":      "Evaluation Agent",
    "debugger":        "Debugger",
    "reviewer":        "Reviewer",
}


def _fmt_duration(ms: Optional[int]) -> str:
    if ms is None:
        return "—"
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms/1000:.1f}s"


def _fmt_time(ts_str: Optional[str]) -> str:
    if not ts_str:
        return "—"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return ts_str[:8]


def _event_icon(event_type: str, status: str) -> str:
    if status == "FAILED":
        return "❌"
    if status == "WARNING":
        return "⚠️"
    if status == "RETRYING":
        return "🔄"
    if status == "FALLBACK":
        return "↩️"
    if status in ("SUCCESS", "RUNNING"):
        if event_type in ("AGENT_COMPLETED", "WORKFLOW_COMPLETED", "TRAINING_COMPLETED",
                           "EVALUATION_COMPLETED", "REVIEW_COMPLETED", "REQUIREMENT_PASSED"):
            return "✅"
        if event_type in ("AGENT_STARTED", "TRAINING_STARTED", "EVALUATION_STARTED",
                           "REVIEW_STARTED", "WORKFLOW_STARTED"):
            return "⟳" if status == "RUNNING" else "✅"
        if event_type == "DECISION":
            return "🧠"
        if event_type in ("TOOL_COMPLETED",):
            return "🔧"
        if event_type == "HANDOFF":
            return "→"
        if event_type in ("LLM_REQUEST", "LLM_RESPONSE"):
            return "🤖"
        if event_type == "LLM_FALLBACK":
            return "↩️"
        if event_type == "LLM_RETRY":
            return "🔄"
        if event_type == "CHECKPOINT_CREATED":
            return "💾"
        if event_type == "DEBUGGER_ANALYSIS":
            return "🔍"
    return "○"


# ─── Main dashboard renderer ─────────────────────────────────────────────────

def render_trace_dashboard(workflow_id: str, api_client, is_running: bool = False, key_prefix: str = ""):
    """
    Renders the full Phase 15 observability dashboard.

    Parameters:
        workflow_id: The workflow to display trace for.
        api_client:  Instance of APIClient from ui.api_client.
        is_running:  If True, the dashboard shows live-polling indicators.
        key_prefix:  Prefix for Streamlit widget keys to prevent conflicts.
    """
    st.markdown(
        "<h2 style='margin-bottom:0'>🔭 Agent Trace</h2>"
        "<p style='color:#aaa;margin-top:0;font-size:0.9em'>Live observability — agent decisions, tool calls, and results.</p>",
        unsafe_allow_html=True
    )

    # Fetch trace data
    trace_data   = api_client.get_trace(workflow_id, limit=200) or {}
    trace_summary = api_client.get_trace_summary(workflow_id)
    events: List[Dict] = trace_data.get("events", [])

    if not events:
        if is_running:
            st.info("⟳ Waiting for trace events... (workflow is running)")
        else:
            st.info("No trace events found for this workflow.")
        return

    # ─── Summary bar ─────────────────────────────────────────────────────────
    if trace_summary:
        _render_summary_bar(trace_summary)

    # ─── Tab layout ──────────────────────────────────────────────────────────
    tab_trace, tab_progress, tab_perf, tab_tools, tab_llm, tab_decisions, tab_errors, tab_export = st.tabs([
        "📋 Trace",
        "📈 Progress",
        "⏱️ Performance",
        "🔧 Tools",
        "🤖 LLM",
        "🧠 Decisions",
        "⚠️ Errors",
        "📤 Export",
    ])

    with tab_trace:
        _render_agent_trace(events, is_running, key_prefix)

    with tab_progress:
        _render_workflow_progress(events)

    with tab_perf:
        _render_agent_performance(trace_summary)

    with tab_tools:
        _render_tool_usage(events)

    with tab_llm:
        _render_llm_usage(trace_summary, events)

    with tab_decisions:
        _render_decision_timeline(events)

    with tab_errors:
        _render_error_timeline(events)

    with tab_export:
        _render_export(workflow_id, api_client, key_prefix)


def _render_summary_bar(summary: Dict):
    cols = st.columns(7)
    cols[0].metric("Status",       summary.get("status", "—"))
    cols[1].metric("Total Events", summary.get("total_events", 0))
    cols[2].metric("Agents",       summary.get("agents", 0))
    cols[3].metric("Retries",      summary.get("retries", 0))
    cols[4].metric("Fallbacks",    summary.get("fallbacks", 0))
    cols[5].metric("Failures",     summary.get("failures", 0))
    cols[6].metric("Duration",     _fmt_duration(summary.get("duration_ms")))


def _render_agent_trace(events: List[Dict], is_running: bool, key_prefix: str):
    """Chronological agent trace list. Shows agent, time, event type, message, duration."""
    st.markdown("### Agent Trace Timeline")
    st.caption("Events are ordered chronologically. Expand any row for details.")

    # Filter controls
    col_agent, col_type, col_status = st.columns(3)
    all_agents = sorted(set(str(e.get("agent_name") or "System") for e in events))
    all_types  = sorted(set(str(e.get("event_type") or "Unknown") for e in events))
    filter_agent  = col_agent.selectbox("Filter by agent",  ["All"] + all_agents, key=f"{key_prefix}tr_agent")
    filter_type   = col_type.selectbox("Filter by type",    ["All"] + all_types,  key=f"{key_prefix}tr_type")
    filter_status = col_status.selectbox("Filter by status", ["All", "SUCCESS", "FAILED", "WARNING", "RUNNING", "RETRYING", "FALLBACK"], key=f"{key_prefix}tr_status")

    filtered = [
        e for e in events
        if (filter_agent == "All" or str(e.get("agent_name") or "System") == filter_agent)
        and (filter_type  == "All" or str(e.get("event_type") or "Unknown") == filter_type)
        and (filter_status == "All" or e.get("status")     == filter_status)
    ]

    if not filtered:
        st.info("No events match the current filter.")
        return

    for i, ev in enumerate(filtered):
        agent   = ev.get("agent_name", "system")
        etype   = ev.get("event_type", "")
        estatus = ev.get("status", "")
        msg     = ev.get("message", "")
        ts      = _fmt_time(ev.get("timestamp"))
        dur     = _fmt_duration(ev.get("duration_ms"))
        icon    = _event_icon(etype, estatus)
        color   = _EVENT_COLOR.get(etype, "#aaaaaa")
        label   = _AGENT_LABELS.get(agent, agent.replace("_", " ").title())

        header = f"{icon} `{ts}` &nbsp; **{label}** &nbsp; <span style='color:{color};font-size:0.85em'>{etype}</span>"
        if dur != "—":
            header += f" &nbsp; ⏱ {dur}"

        with st.expander(f"{icon}  {ts}  {label} — {msg[:80]}", expanded=False):
            st.markdown(header, unsafe_allow_html=True)
            st.write(f"**Message:** {msg}")
            if ev.get("decision"):
                st.write(f"**Decision:** {ev['decision']}")
            if ev.get("action"):
                st.write(f"**Action:** {ev['action']}")
            if ev.get("tool_name"):
                st.write(f"**Tool:** {ev['tool_name']}")
            if ev.get("result_summary"):
                st.write(f"**Result:** {ev['result_summary']}")
            if ev.get("provider"):
                st.write(f"**Provider:** {ev['provider']}")
            if ev.get("error_code"):
                st.error(f"Error: {ev['error_code']}")
            if ev.get("metadata"):
                with st.expander("Metadata"):
                    st.json(ev["metadata"])
            st.caption(f"Event ID: {ev.get('event_id', '—')} | Trace: {ev.get('trace_id', '—')}")

    if is_running:
        st.info("⟳ Workflow is running — trace updates automatically.")


def _render_workflow_progress(events: List[Dict]):
    """Visual workflow progress bar based on actual agent completion events."""
    st.markdown("### Workflow Progress")

    agent_status: Dict[str, str] = {}
    for ev in events:
        agent = ev.get("agent_name", "")
        etype = ev.get("event_type", "")
        if etype == "AGENT_STARTED":
            if agent not in agent_status:
                agent_status[agent] = "RUNNING"
        elif etype == "AGENT_COMPLETED":
            agent_status[agent] = "SUCCESS"
        elif etype == "AGENT_FAILED":
            agent_status[agent] = "FAILED"

    cols = st.columns(len(_AGENT_WORKFLOW_ORDER))
    for i, agent in enumerate(_AGENT_WORKFLOW_ORDER):
        label = _AGENT_LABELS.get(agent, agent)
        status = agent_status.get(agent)
        icon = _STATUS_ICON.get(status, "○")
        color = "#2ecc71" if status == "SUCCESS" else "#e74c3c" if status == "FAILED" else "#f39c12" if status == "RUNNING" else "#555"
        cols[i].markdown(
            f"<div style='text-align:center;padding:8px;border:1px solid {color};border-radius:6px;"
            f"background:rgba(0,0,0,0.2);margin:2px'>"
            f"<div style='font-size:1.4em'>{icon}</div>"
            f"<div style='font-size:0.75em;color:{color};margin-top:4px'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Current agent
    running_agents = [a for a, s in agent_status.items() if s == "RUNNING"]
    if running_agents:
        current = _AGENT_LABELS.get(running_agents[-1], running_agents[-1])
        st.markdown(
            f"<div style='margin-top:16px;padding:12px;background:#1a2a1a;border-radius:8px;"
            f"border:1px solid #2ecc71'>"
            f"<b>Current Agent:</b> {current} &nbsp; <span style='color:#f39c12'>⟳ RUNNING</span>"
            f"</div>",
            unsafe_allow_html=True
        )


def _render_agent_performance(summary: Optional[Dict]):
    """Agent performance table with durations."""
    st.markdown("### Agent Performance")
    if not summary:
        st.info("No performance data available.")
        return

    durations = summary.get("agent_durations", {})
    if not durations:
        st.info("No duration data recorded.")
        return

    rows = []
    for agent, dur_ms in durations.items():
        agent_str = str(agent) if agent else "system"
        label = _AGENT_LABELS.get(agent_str, agent_str.replace("_", " ").title())
        rows.append({"Agent": label, "Duration": _fmt_duration(dur_ms), "Duration (ms)": dur_ms or 0})

    rows.sort(key=lambda r: r["Duration (ms)"], reverse=True)

    import pandas as pd
    df = pd.DataFrame(rows)[["Agent", "Duration"]]
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Total workflow duration
    wf_dur = summary.get("duration_ms")
    if wf_dur:
        st.caption(f"⏱ Total workflow duration: **{_fmt_duration(wf_dur)}**")


def _render_tool_usage(events: List[Dict]):
    """Tool usage statistics."""
    st.markdown("### Tool Usage")
    tool_counts: Dict[str, int] = {}
    tool_failures: Dict[str, int] = {}
    for ev in events:
        tool = ev.get("tool_name")
        if not tool:
            continue
        etype = ev.get("event_type", "")
        if etype == "TOOL_COMPLETED":
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        elif etype == "TOOL_FAILED":
            tool_failures[tool] = tool_failures.get(tool, 0) + 1

    if not tool_counts and not tool_failures:
        st.info("No tool usage data recorded.")
        return

    all_tools = set(list(tool_counts.keys()) + list(tool_failures.keys()))
    rows = [
        {
            "Tool": t,
            "Successes": tool_counts.get(t, 0),
            "Failures": tool_failures.get(t, 0),
        }
        for t in sorted(all_tools)
    ]
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_llm_usage(summary: Optional[Dict], events: List[Dict]):
    """LLM provider usage breakdown."""
    st.markdown("### LLM Usage")

    if summary:
        provider_usage = summary.get("provider_usage", {})
    else:
        provider_usage = {}

    # Derive from events if summary unavailable
    if not provider_usage:
        for ev in events:
            p = ev.get("provider")
            if not p:
                continue
            if p not in provider_usage:
                provider_usage[p] = {"calls": 0, "successes": 0, "failures": 0, "fallbacks": 0}
            if ev.get("event_type") == "LLM_RESPONSE":
                provider_usage[p]["calls"]     += 1
                provider_usage[p]["successes"] += 1
            elif ev.get("event_type") == "LLM_FALLBACK":
                provider_usage[p]["fallbacks"] += 1

    if not provider_usage:
        st.info("No LLM usage data recorded.")
        return

    cols = st.columns(max(len(provider_usage), 1))
    for i, (prov, stats) in enumerate(provider_usage.items()):
        with cols[i]:
            st.markdown(f"**{prov.title()}**")
            st.metric("Calls",     stats.get("calls", 0))
            st.metric("Successes", stats.get("successes", 0))
            st.metric("Failures",  stats.get("failures", 0))
            if stats.get("fallbacks"):
                st.metric("Fallbacks", stats["fallbacks"])

    # Retry / fallback counts
    if summary:
        c1, c2 = st.columns(2)
        c1.metric("Total Retries",   summary.get("retries", 0))
        c2.metric("Total Fallbacks", summary.get("fallbacks", 0))


def _render_decision_timeline(events: List[Dict]):
    """Decision events only — structured, no chain-of-thought."""
    st.markdown("### Decision Timeline")
    decisions = [e for e in events if e.get("event_type") == "DECISION"]
    if not decisions:
        st.info("No decision events recorded.")
        return

    for ev in decisions:
        agent = _AGENT_LABELS.get(ev.get("agent_name", ""), ev.get("agent_name", ""))
        ts    = _fmt_time(ev.get("timestamp"))
        dec   = ev.get("decision", "—")
        msg   = ev.get("message", "—")
        meta  = ev.get("metadata") or {}

        st.markdown(
            f"<div style='padding:10px;margin:6px 0;border-left:3px solid #9b59b6;"
            f"background:#1a1a2e;border-radius:0 6px 6px 0'>"
            f"<span style='color:#aaa;font-size:0.8em'>{ts}</span> &nbsp;"
            f"<b style='color:#9b59b6'>{agent}</b> &nbsp; "
            f"<span style='color:#fff'>{msg}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        if meta:
            with st.expander("Details", expanded=False):
                st.json(meta)


def _render_error_timeline(events: List[Dict]):
    """Error and failure events timeline."""
    st.markdown("### Error & Failure Timeline")
    errors = [
        e for e in events
        if e.get("status") in ("FAILED", "WARNING")
        or e.get("event_type") in ("ERROR", "REQUIREMENT_FAILED", "TOOL_FAILED", "AGENT_FAILED", "WORKFLOW_FAILED")
    ]

    if not errors:
        st.success("✅ No errors or warnings recorded.")
        return

    for ev in errors:
        agent  = _AGENT_LABELS.get(ev.get("agent_name", ""), ev.get("agent_name", ""))
        ts     = _fmt_time(ev.get("timestamp"))
        msg    = ev.get("message", "—")
        etype  = ev.get("event_type", "")
        is_err = ev.get("status") == "FAILED"
        icon   = "❌" if is_err else "⚠️"

        st.markdown(
            f"<div style='padding:10px;margin:6px 0;border-left:3px solid {'#e74c3c' if is_err else '#f39c12'};"
            f"background:#1a1a2e;border-radius:0 6px 6px 0'>"
            f"<span style='color:#aaa;font-size:0.8em'>{ts}</span> &nbsp;"
            f"{icon} <b>{agent}</b> — {msg}"
            f"</div>",
            unsafe_allow_html=True
        )


def _render_export(workflow_id: str, api_client, key_prefix: str):
    """JSON trace export button."""
    st.markdown("### Export Trace")
    st.markdown(
        "Download the complete workflow trace as JSON. "
        "Safe: no chain-of-thought, API keys, raw prompts, or secrets included."
    )
    if st.button("📥 Download Trace JSON", key=f"{key_prefix}btn_export_trace"):
        try:
            data = api_client.export_trace(workflow_id)
            st.download_button(
                label="💾 Save trace JSON",
                data=data,
                file_name=f"trace_{workflow_id}.json",
                mime="application/json",
                key=f"{key_prefix}btn_download_trace"
            )
        except Exception as e:
            st.error(f"Export failed: {e}")
