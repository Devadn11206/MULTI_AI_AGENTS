import streamlit as st
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui.api_client import APIClient
from ui.trace_dashboard import render_trace_dashboard

st.set_page_config(page_title="MLForge", layout="wide", page_icon="⚒️")

if "workflow_id" not in st.session_state:
    st.session_state["workflow_id"] = None
if "dataset_id" not in st.session_state:
    st.session_state["dataset_id"] = None
if "result" not in st.session_state:
    st.session_state["result"] = None
if "columns_info" not in st.session_state:
    st.session_state["columns_info"] = None
if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None
if "_last_metric" not in st.session_state:
    st.session_state["_last_metric"] = "auto"


def status_badge(s):
    colors = {
        "completed": "green", "success": "green", "pass": "green",
        "failed": "red", "fail": "red",
        "running": "orange", "queued": "gray",
        "pass_with_warnings": "#b8860b"
    }
    c = colors.get(str(s).lower(), "gray")
    return f"<span style='background:{c};color:white;padding:2px 10px;border-radius:4px;font-size:0.85em'>{str(s).upper()}</span>"


def metric_card(label, value):
    try:
        display = f"{float(value):.4f}"
    except Exception:
        display = str(value)
    return (
        f"<div style='background:#1e1e2e;border-radius:8px;padding:16px 20px;"
        f"margin:6px 0;border:1px solid #333'>"
        f"<div style='color:#aaa;font-size:0.8em;margin-bottom:4px'>{label}</div>"
        f"<div style='color:#fff;font-size:1.5em;font-weight:bold'>{display}</div></div>"
    )


def render_best_model(best):
    if not best:
        st.info("No model metrics available.")
        return
    cols = st.columns(min(len(best), 5))
    for i, (k, v) in enumerate(best.items()):
        with cols[i % len(cols)]:
            st.markdown(metric_card(k.upper(), v), unsafe_allow_html=True)


def render_dataset_tab(result):
    info = result.get("dataset_info") or {}
    profile = result.get("dataset_profile") or {}
    st.markdown("### Dataset Info")
    c1, c2, c3 = st.columns(3)
    filename = info.get("original_filename") or info.get("filename") or os.path.basename(str(info.get("path", "N/A")))
    c1.metric("File", filename)
    c2.metric("Target Column", info.get("target_column") or "N/A")
    c3.metric("Task Type", result.get("task_type") or "Auto")
    if profile:
        st.markdown("---")
        st.markdown("### Feature Engineering Profile")
        for key, val in profile.items():
            if isinstance(val, dict):
                with st.expander(f"**{key.replace('_', ' ').title()}**"):
                    st.json(val)
            elif isinstance(val, list):
                with st.expander(f"**{key.replace('_', ' ').title()}** ({len(val)} items)"):
                    st.json(val)
            else:
                st.write(f"**{key.replace('_', ' ').title()}:** {val}")
    else:
        st.info("No dataset profile available.")


def render_experiments_tab(result):
    training = result.get("training_results") or []
    evaluation = result.get("evaluation_results") or []
    if not training and not evaluation:
        st.info("No training or evaluation data available.")
        return
    st.markdown("### Training Results")
    for i, tr in enumerate(training):
        model = tr.get("model_name") or tr.get("model_id") or f"Model {i+1}"
        status = tr.get("status", "unknown")
        icon = "✅" if status == "completed" else "❌"
        with st.expander(f"{icon} {model}"):
            c1, c2 = st.columns(2)
            c1.write(f"**Status:** {status}")
            c1.write(f"**Time:** {tr.get('training_time_seconds', 'N/A')}s")
            if tr.get("train_metrics"):
                c2.write("**Train Metrics:**")
                for k, v in tr["train_metrics"].items():
                    c2.write(f"  - {k}: `{v:.4f}`")
            if tr.get("error"):
                st.error(f"Error: {tr['error']}")
    if evaluation:
        st.markdown("---")
        st.markdown("### Evaluation Metrics")
        for i, ev in enumerate(evaluation):
            metrics = ev.get("metrics") or {}
            model = ev.get("model_name") or ev.get("model_id") or f"Model {i+1}"
            with st.expander(f"📈 {model}"):
                if metrics:
                    mcols = st.columns(min(len(metrics), 5))
                    for j, (k, v) in enumerate(metrics.items()):
                        mcols[j % len(mcols)].metric(k.upper(), f"{float(v):.4f}")
                else:
                    st.info("No metrics recorded.")


def render_correction_tab(result):
    history = result.get("self_correction_history") or []
    if not history:
        st.info("No self-correction attempts recorded.")
        return
    st.markdown(f"### Self-Correction History ({len(history)} attempt(s))")
    for attempt in history:
        num = attempt.get("attempt_number", "?")
        status = attempt.get("status", "unknown")
        icon = "✅" if status == "pass" else "❌"
        with st.expander(f"{icon} Attempt {num} — {status.upper()}"):
            req_results = attempt.get("requirement_result") or []
            if req_results:
                st.markdown("**Requirement Results:**")
                for rr in req_results:
                    s = rr.get("status", "?")
                    st.write(f"{'✅' if s == 'pass' else '❌'} {rr.get('metric', 'metric')} — {rr.get('message', '')}")
            ev = attempt.get("evaluation_results") or {}
            metrics = ev.get("metrics") or {}
            if metrics:
                st.markdown("**Metrics:**")
                mcols = st.columns(min(len(metrics), 5))
                for j, (k, v) in enumerate(metrics.items()):
                    try:
                        mcols[j % len(mcols)].metric(k.upper(), f"{float(v):.4f}")
                    except Exception:
                        mcols[j % len(mcols)].metric(k.upper(), str(v))
            action = attempt.get("action")
            if action:
                st.info(f"**Corrective Action Applied:** {action}")


def render_review_tab(result):
    rev = result.get("reviewer_result") or {}
    if not rev:
        st.info("No review data available.")
        return
    status = rev.get("status", "N/A")
    st.markdown(f"### Final Review — {status_badge(status)}", unsafe_allow_html=True)
    st.markdown("---")
    summary = rev.get("reviewer_summary") or rev.get("summary")
    if summary:
        st.markdown("**Reviewer Summary:**")
        st.info(summary)
    checks = rev.get("checks") or []
    if checks:
        st.markdown("**Detailed Checks:**")
        for check in checks:
            s = check.get("status", "?")
            icon = "✅" if s == "PASS" else "⚠️" if s == "WARNING" else "❌"
            with st.expander(f"{icon} [{check.get('category','?')}] {check.get('name','?')} — {s}"):
                st.write(f"**Severity:** {check.get('severity', 'N/A')}")
                st.write(f"**Evidence:** {check.get('evidence', 'N/A')}")
                
    usage = result.get("usage_stats")
    if usage:
        st.markdown("---")
        st.markdown("### Reliability / Usage")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gemini Calls", usage.get("gemini", {}).get("calls", 0))
        c2.metric("Groq Calls", usage.get("groq", {}).get("calls", 0))
        c3.metric("Fallbacks", usage.get("fallbacks", 0))
        
        gemini_fails = usage.get("gemini", {}).get("failures", 0)
        c4.metric("Provider Failures", gemini_fails + usage.get("groq", {}).get("failures", 0))



# ─── Metric options per task type ────────────────────────────────────────────
CLASSIFICATION_METRICS = ["recall", "accuracy", "precision", "f1", "roc_auc"]
REGRESSION_METRICS = ["rmse", "mae", "mse", "r2"]
ALL_METRICS = ["auto"] + CLASSIFICATION_METRICS + REGRESSION_METRICS

METRIC_DEFAULTS = {
    "classification": "recall",
    "regression": "rmse",
    "auto": "auto",
}

# Known ID-like column patterns
_ID_PATTERNS = {"id", "customerid", "customer_id", "property_id", "transaction_id",
                "userid", "user_id", "rowid", "row_id", "index"}

# Heuristic target names (suggest, never force)
_TARGET_HINTS = {"target", "label", "class", "outcome", "prediction",
                 "price", "churn", "fraud", "default", "survived", "y",
                 "sales", "revenue", "demand", "rating"}


def _suggest_target(columns):
    """Return a best-guess target column name or None."""
    lower_map = {c.lower(): c for c in columns}
    for hint in _TARGET_HINTS:
        if hint in lower_map:
            return lower_map[hint]
    # Last column fallback
    return columns[-1] if columns else None


def _is_id_column(col_name: str) -> bool:
    return col_name.lower() in _ID_PATTERNS or col_name.lower().endswith("_id")


def _get_metrics_for_task(task_type: str):
    if task_type == "classification":
        return CLASSIFICATION_METRICS
    elif task_type == "regression":
        return REGRESSION_METRICS
    return ALL_METRICS


def _safe_default_metric(task_type: str, current_metric: str) -> str:
    """If current metric is invalid for the task type, reset to the default."""
    valid = _get_metrics_for_task(task_type)
    if current_metric in valid:
        return current_metric
    return METRIC_DEFAULTS[task_type]


def main():
    st.markdown("<h1 style='font-size:2.5em'>⚒️ MLForge</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#aaa;margin-top:-10px'>Multi-Agent ML Engineering Platform</p>",
                unsafe_allow_html=True)

    # Auth
    if "access_token" not in st.session_state:
        st.session_state["access_token"] = None

    if not st.session_state["access_token"]:
        st.warning("Please login to access MLForge.")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                st.subheader("Login")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    try:
                        res = APIClient.login(username, password)
                        st.session_state["access_token"] = res["access_token"]
                        st.success("Logged in successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: Incorrect username or password")
                        
        with tab2:
            with st.form("register_form"):
                st.subheader("Register")
                reg_username = st.text_input("Username")
                reg_email = st.text_input("Email")
                reg_password = st.text_input("Password", type="password")
                if st.form_submit_button("Register"):
                    try:
                        APIClient.register(reg_username, reg_email, reg_password)
                        st.success("Registered successfully! Please login.")
                    except Exception as e:
                        st.error(f"Registration failed: {e}")
        st.stop()
        
    st.sidebar.button("Logout", on_click=lambda: st.session_state.pop("access_token", None))

    # ─── Sidebar ─────────────────────────────────────────────────────────────
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Mode", ["Studio", "Evaluation"])
    
    if app_mode == "Evaluation":
        from ui.evaluation_dashboard import render_evaluation_dashboard
        render_evaluation_dashboard()
        return
        
    st.sidebar.markdown("---")
    st.sidebar.title("System Status")
    health = APIClient.get_health()
    if health:
        st.sidebar.success("✅ Backend: Connected")
        prov = APIClient.get_providers()
        if prov:
            st.sidebar.markdown("### LLM Providers")
            
            gemini_stat = prov.get('gemini', {})
            groq_stat = prov.get('groq', {})
            
            if gemini_stat.get("status") == "AVAILABLE":
                st.sidebar.markdown("**Gemini** ✓ Available")
            elif gemini_stat.get("status") == "UNAVAILABLE" or not gemini_stat.get("configured"):
                st.sidebar.markdown("**Gemini** ✗ Unavailable")
            else:
                st.sidebar.markdown(f"**Gemini** ⚠ {gemini_stat.get('status').replace('_', ' ').title()}")
                
            if groq_stat.get("status") == "AVAILABLE":
                st.sidebar.markdown("**Groq** ✓ Available")
            elif groq_stat.get("status") == "UNAVAILABLE" or not groq_stat.get("configured"):
                st.sidebar.markdown("**Groq** ✗ Unavailable")
            else:
                st.sidebar.markdown(f"**Groq** ⚠ {groq_stat.get('status').replace('_', ' ').title()}")
                
        # Recovery Section
        interrupted = APIClient.get_recent_interrupted()
        if interrupted:
            st.sidebar.markdown("---")
            st.sidebar.warning("⚠️ Workflows Interrupted")
            for wf in interrupted[:3]:
                with st.sidebar.expander(f"Recover {wf.get('workflow_id')}"):
                    st.write(f"**Latest Checkpoint:** {wf.get('phase').title()}")
                    if st.button("Resume Workflow", key=f"resume_{wf.get('workflow_id')}"):
                        APIClient.resume_workflow(wf.get("workflow_id"))
                        st.session_state["workflow_id"] = wf.get("workflow_id")
                        st.session_state["result"] = None
                        st.rerun()

                        
    else:
        st.sidebar.error("❌ Backend: Unavailable")
        st.error("MLForge backend is unavailable. Run: `uvicorn api.main:app --reload`")
        return

    # ─── Step 1: Dataset Upload ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 1. Dataset")

    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV / XLSX)", type=["csv", "xlsx"],
        help="Upload the dataset you want MLForge to model."
    )

    if uploaded_file:
        # Detect if a new file was uploaded (name changed) → reset state
        if uploaded_file.name != st.session_state.get("uploaded_filename"):
            st.session_state["dataset_id"] = None
            st.session_state["columns_info"] = None
            st.session_state["result"] = None
            st.session_state["workflow_id"] = None
            st.session_state["uploaded_filename"] = uploaded_file.name

        if st.button("📤 Upload Dataset", type="secondary", key="btn_upload"):
            with st.spinner("Uploading..."):
                try:
                    res = APIClient.upload_dataset(uploaded_file, uploaded_file.name)
                    st.session_state["dataset_id"] = res["dataset_id"]
                    # Fetch column info immediately after upload
                    col_info = APIClient.get_columns(res["dataset_id"])
                    st.session_state["columns_info"] = col_info
                    st.session_state["result"] = None
                    st.session_state["workflow_id"] = None
                    st.success(f"✅ Uploaded! Dataset ID: `{res['dataset_id']}`")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    # Show dataset preview if we have column info
    col_info = st.session_state.get("columns_info")
    if col_info:
        with st.expander("📊 Dataset Preview", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", col_info.get("row_count", "?"))
            c2.metric("Columns", col_info.get("col_count", "?"))
            c3.metric("Missing Values", col_info.get("missing_values", "?"))
            preview = col_info.get("preview", [])
            if preview:
                import pandas as pd
                st.dataframe(pd.DataFrame(preview), width="stretch")

    if not st.session_state.get("dataset_id") and not st.session_state.get("workflow_id"):
        st.info("⬆️ Upload a dataset to begin.")
        return


    # Steps 2 to 6 are for configuring a new workflow run
    if st.session_state.get("dataset_id"):
        # ─── Step 2: Objective ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 2. Objective")
        objective = st.text_input(
            "What should MLForge predict?",
            placeholder="e.g. Predict customer churn",
            key="input_objective"
        )

        # ─── Step 3: Task Type ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 3. Task Type")
        task_type = st.selectbox(
            "Task Type",
            options=["auto", "classification", "regression"],
            index=0,
            format_func=lambda x: x.capitalize(),
            key="input_task_type",
            help="Choose Auto to let MLForge determine the ML problem type automatically."
        )

        # ─── Step 4: Target Column ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 4. Target Column")

        columns = col_info.get("columns", []) if col_info else []
        dtypes = col_info.get("dtypes", {}) if col_info else {}

        if not columns:
            st.error("No columns found in dataset. Please re-upload a valid CSV.")
            return

        # Auto-suggest target
        suggested = _suggest_target(columns)
        default_idx = columns.index(suggested) if suggested in columns else len(columns) - 1

        target_column = st.selectbox(
            "Target Column",
            options=columns,
            index=default_idx,
            key="input_target_col",
            help="Select the variable MLForge should predict."
        )

        # Auto-suggestion annotation
        if suggested and target_column == suggested:
            st.caption(f"💡 Auto-suggested based on column name heuristics: **{suggested}**")

        # ID-column warning
        if _is_id_column(target_column):
            st.warning(
                f"⚠️ **'{target_column}'** appears to be an identifier column. "
                "Using it as the target may produce an invalid ML task. You can still proceed."
            )

        # Regression + categorical target warning
        col_dtype = dtypes.get(target_column, "")
        target_is_numeric = "int" in col_dtype or "float" in col_dtype
        if task_type == "regression" and not target_is_numeric:
            st.error(
                f"❌ Regression requires a **numerical target column**, "
                f"but **'{target_column}'** has dtype `{col_dtype}`. "
                "Please choose a numeric column or change the Task Type."
            )
            # Block workflow creation
            return

        # ─── Step 5: Primary Metric ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 5. Primary Metric")

        valid_metrics = _get_metrics_for_task(task_type)
        prev_metric = st.session_state.get("_last_metric", METRIC_DEFAULTS[task_type])
        safe_metric = _safe_default_metric(task_type, prev_metric)
        metric_idx = valid_metrics.index(safe_metric) if safe_metric in valid_metrics else 0

        priority = st.selectbox(
            "Primary Metric",
            options=valid_metrics,
            index=metric_idx,
            format_func=lambda x: x.upper() if x != "auto" else "Auto",
            key=f"input_priority_{task_type}",
            help="Metric used to prioritize model selection and evaluation."
        )
        st.session_state["_last_metric"] = priority

        if task_type == "classification":
            st.caption("Classification metrics: Accuracy · Precision · Recall · F1 · ROC-AUC")
        elif task_type == "regression":
            st.caption("Regression metrics: RMSE · MAE · MSE · R²")
        else:
            st.caption("Auto: MLForge selects the appropriate metric after determining the task type.")

        # ─── Step 6: RUN MLFORGE ─────────────────────────────────────────────────
        st.markdown("---")
        run_disabled = not objective or not target_column
        if run_disabled:
            st.caption("⬅️ Fill in the Objective and select a Target Column to begin.")

        if st.button("🚀 RUN MLFORGE", disabled=run_disabled, type="primary", key="btn_run"):
            payload_log = {
                "dataset_id": st.session_state["dataset_id"],
                "objective": objective,
                "task_type": task_type,
                "target_column": target_column,
                "priority": priority,
            }
            st.caption(f"📋 Request payload: `{payload_log}`")
            try:
                res = APIClient.create_workflow(
                    dataset_id=st.session_state["dataset_id"],
                    objective=objective,
                    priority=priority,
                    target_column=target_column,
                    task_type=task_type,
                )
                st.session_state["workflow_id"] = res["workflow_id"]
                st.session_state["result"] = None
                st.success(f"✅ Workflow started! ID: `{res['workflow_id']}`")
            except Exception as e:
                st.error(f"Failed to start workflow: {e}")
    else:
        task_type = "classification"


    # ─── Monitoring ──────────────────────────────────────────────────────────
    if st.session_state.get("workflow_id"):
        st.markdown("---")
        wf_id = st.session_state["workflow_id"]
        st.markdown(f"## 📡 Workflow `{wf_id}`")
        status_col, activity_col = st.columns([1, 2])
        with status_col:
            st.markdown("### Status")
            status_container = st.empty()
            progress_bar = st.progress(0)
        with activity_col:
            st.markdown("### Agent Activity")
            activity_container = st.empty()

        while True:
            try:
                wf = APIClient.get_workflow_status(wf_id)
                events = APIClient.get_events(wf_id)
            except Exception:
                break
            with status_container.container():
                st.markdown(status_badge(wf["status"]), unsafe_allow_html=True)
                st.write(f"**Agent:** {wf.get('current_agent', '—')}")
                st.caption(wf.get("message", ""))
            progress_bar.progress(wf["progress"] / 100.0)
            with activity_container.container():
                for event in events:
                    s = event["status"]
                    icon = "✅" if s == "completed" else "❌" if s == "failed" else "⏳"
                    st.write(f"{icon} **{event['agent'].replace('_', ' ').title()}**: {event['message']}")
            if wf["status"] in ["completed", "failed", "cancelled"]:
                break
            time.sleep(2)
            st.rerun()

        if st.session_state.get("result") is None:
            st.session_state["result"] = APIClient.get_result(wf_id)

        # ─ Phase 15: Show live trace below monitor ─
        st.markdown("---")
        st.markdown("### 🔭 Live Trace")
        render_trace_dashboard(wf_id, APIClient, is_running=False, key_prefix="live_")

    # ─── Results ─────────────────────────────────────────────────────────────
    result = st.session_state.get("result")
    if result:
        st.markdown("---")
        st.markdown("## 📋 Results")

        # Show detected task type from backend (authoritative)
        detected_task = result.get("task_type") or task_type
        st.markdown(
            f"**Task Type (detected):** {status_badge(detected_task.upper())}",
            unsafe_allow_html=True
        )

        best = result.get("best_model") or {}
        if best:
            st.markdown("### 🏆 Best Model Metrics")
            # Filter metrics by detected task type
            if detected_task == "regression":
                show_keys = REGRESSION_METRICS
            elif detected_task == "classification":
                show_keys = CLASSIFICATION_METRICS
            else:
                show_keys = list(best.keys())
            filtered = {k: v for k, v in best.items() if k.lower() in show_keys or k.lower() in list(best.keys())}
            render_best_model(filtered or best)

        errors = result.get("errors") or []
        warnings = result.get("warnings") or []
        if errors or warnings:
            st.markdown("### Errors & Warnings")
            for err in errors:
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                st.error(f"❌ {msg}")
            for w in warnings:
                st.warning(f"⚠️ {w}")

        st.markdown("---")
        tabs = st.tabs(["📂 Dataset", "🧪 Experiments", "🔄 Self-Correction", "🔍 Review", "🔭 Trace"])
        with tabs[0]:
            render_dataset_tab(result)
        with tabs[1]:
            render_experiments_tab(result)
        with tabs[2]:
            render_correction_tab(result)
        with tabs[3]:
            render_review_tab(result)
        with tabs[4]:
            wf_id = st.session_state.get("workflow_id")
            if wf_id:
                render_trace_dashboard(wf_id, APIClient, is_running=False, key_prefix="res_")
            else:
                st.info("No workflow ID found.")


if __name__ == "__main__":
    main()

