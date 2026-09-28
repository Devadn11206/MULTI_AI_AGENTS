import streamlit as st
import pandas as pd
from ui.api_client import APIClient

def render_evaluation_dashboard():
    st.markdown("## 🧪 MLForge Evaluation & Testing")
    
    # Run evaluation section
    st.markdown("### Run New Evaluation")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        mode = st.selectbox("Evaluation Mode", ["mock", "live"], index=0, help="Mock runs fast failures, Live tests actual LLMs and workflows.")
    with col2:
        test_type = st.selectbox("Test Type", ["all", "unit", "workflows"], index=0)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶ Run Evaluation", type="primary"):
            st.info("Evaluation triggered. In a production setting, this would start the background job.")
            
    st.markdown("---")
    st.markdown("### Evaluation History")
    
    try:
        # We would normally get from API, but we can hit DB directly or through requests
        import requests
        res = requests.get(f"{APIClient.BASE_URL}/evaluations")
        runs = res.json() if res.status_code == 200 else []
    except Exception as e:
        st.error(f"Failed to fetch evaluations: {e}")
        runs = []
        
    if not runs:
        st.info("No evaluation runs found.")
        return
        
    # Latest Run Summary
    latest = runs[0]
    st.markdown(f"**Latest Run ID:** `{latest.get('evaluation_run_id')}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", latest.get("mode", "unknown").upper())
    c2.metric("Total Tests", latest.get("total_tests", 0))
    c3.metric("Passed", latest.get("passed_tests", 0))
    
    success_rate = 0
    if latest.get("total_tests", 0) > 0:
        success_rate = (latest.get("passed_tests", 0) / latest.get("total_tests", 1)) * 100
    c4.metric("Success Rate", f"{success_rate:.1f}%")
    
    st.markdown(f"**Duration:** {latest.get('duration_ms', 0) / 1000:.1f}s")
    
    # Test Matrix
    st.markdown("### Test Results Matrix")
    
    results = latest.get("results", [])
    if results:
        df = pd.DataFrame(results)
        # Format for UI
        df_ui = pd.DataFrame({
            "Test ID": df.get("test_id", ""),
            "Dataset": df.get("dataset_id", ""),
            "Status": df.get("status", "").apply(lambda x: "✅ PASS" if x == "PASS" else ("❌ FAIL" if x == "FAIL" else x)),
            "Time (s)": df.get("duration_ms", 0) / 1000,
            "Error Code": df.get("error_code", "")
        })
        st.dataframe(df_ui, use_container_width=True)
        
        failures = [r for r in results if r.get("status") != "PASS"]
        if failures:
            st.markdown("### Failure Analysis")
            for f in failures:
                with st.expander(f"❌ {f.get('test_id')} - {f.get('error_code')}"):
                    st.write(f"**Dataset:** {f.get('dataset_id')}")
                    st.json(f.get("metadata_json", {}))
    else:
        st.info("No results found in this run.")
        
    # Historical Trend
    if len(runs) > 1:
        st.markdown("### Reliability Trend")
        trend_data = []
        for r in reversed(runs[:10]):
            sr = 0
            if r.get("total_tests", 0) > 0:
                sr = (r.get("passed_tests", 0) / r.get("total_tests", 1)) * 100
            trend_data.append({
                "Run ID": r.get("evaluation_run_id")[:13],
                "Success Rate (%)": sr
            })
        st.line_chart(pd.DataFrame(trend_data).set_index("Run ID"))
