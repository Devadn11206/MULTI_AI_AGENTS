import requests
from ui.config import API_URL
import streamlit as st

class APIClient:
    @staticmethod
    def _get_headers():
        headers = {}
        if "access_token" in st.session_state:
            headers["Authorization"] = f"Bearer {st.session_state['access_token']}"
        return headers

    @staticmethod
    def login(username, password):
        url = f"{API_URL}/auth/login"
        r = requests.post(url, data={"username": username, "password": password})
        r.raise_for_status()
        return r.json()

    @staticmethod
    def register(username, email, password):
        url = f"{API_URL}/auth/register"
        r = requests.post(url, json={"username": username, "email": email, "password": password})
        r.raise_for_status()
        return r.json()

    @staticmethod
    def get_health():
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    @staticmethod
    def get_providers():
        try:
            r = requests.get(f"{API_URL}/health/providers", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    @staticmethod
    def upload_dataset(file_obj, filename: str):
        url = f"{API_URL}/datasets/upload"
        files = {"file": (filename, file_obj, "text/csv")}
        r = requests.post(url, files=files, headers=APIClient._get_headers())
        r.raise_for_status()
        return r.json()

    @staticmethod
    def get_columns(dataset_id: str):
        """Fetch column names, dtypes, shape and preview for a dataset."""
        url = f"{API_URL}/datasets/{dataset_id}/columns"
        try:
            r = requests.get(url, timeout=10, headers=APIClient._get_headers())
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return None

    @staticmethod
    def create_workflow(
        dataset_id: str,
        objective: str,
        priority: str,
        target_column: str = None,
        task_type: str = "auto",
        target_threshold: float = None
    ):
        url = f"{API_URL}/workflows"
        payload = {
            "dataset_id": dataset_id,
            "objective": objective,
            "priority": priority,
            "target_column": target_column,
            "task_type": task_type,
            "target_threshold": target_threshold,
        }
        r = requests.post(url, json=payload, headers=APIClient._get_headers())
        r.raise_for_status()
        return r.json()

    @staticmethod
    def get_workflow_status(workflow_id: str):
        url = f"{API_URL}/workflows/{workflow_id}"
        r = requests.get(url, headers=APIClient._get_headers())
        r.raise_for_status()
        return r.json()

    @staticmethod
    def get_events(workflow_id: str):
        url = f"{API_URL}/workflows/{workflow_id}/events"
        r = requests.get(url, headers=APIClient._get_headers())
        r.raise_for_status()
        return r.json().get("events", [])

    @staticmethod
    def get_result(workflow_id: str):
        url = f"{API_URL}/workflows/{workflow_id}/result"
        r = requests.get(url, headers=APIClient._get_headers())
        if r.status_code == 200:
            return r.json()
        return None

    @staticmethod
    def get_recovery_status(workflow_id: str):
        url = f"{API_URL}/workflows/{workflow_id}/recovery"
        r = requests.get(url, headers=APIClient._get_headers())
        if r.status_code == 200:
            return r.json()
        return None
        
    @staticmethod
    def get_recent_interrupted():
        url = f"{API_URL}/workflows/recent/interrupted"
        r = requests.get(url, headers=APIClient._get_headers())
        if r.status_code == 200:
            return r.json().get("workflows", [])
        return []

    @staticmethod
    def resume_workflow(workflow_id: str):
        url = f"{API_URL}/workflows/{workflow_id}/resume"
        r = requests.post(url, headers=APIClient._get_headers())
        r.raise_for_status()
        return r.json()

    # ── Phase 15: Observability Trace API ─────────────────────────────

    @staticmethod
    def get_trace(
        workflow_id: str,
        agent: str = None,
        event_type: str = None,
        status: str = None,
        limit: int = 200,
        offset: int = 0,
    ):
        url = f"{API_URL}/workflows/{workflow_id}/trace"
        params = {"limit": limit, "offset": offset}
        if agent:      params["agent"]      = agent
        if event_type: params["event_type"] = event_type
        if status:     params["status"]     = status
        try:
            r = requests.get(url, params=params, headers=APIClient._get_headers(), timeout=15)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    @staticmethod
    def get_trace_summary(workflow_id: str):
        url = f"{API_URL}/workflows/{workflow_id}/trace/summary"
        try:
            r = requests.get(url, headers=APIClient._get_headers(), timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    @staticmethod
    def export_trace(workflow_id: str) -> bytes:
        url = f"{API_URL}/workflows/{workflow_id}/trace/export"
        r = requests.get(url, headers=APIClient._get_headers(), timeout=15)
        r.raise_for_status()
        return r.content

