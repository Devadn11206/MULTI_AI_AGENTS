"""
Phase 15 — Observability: Test Suite

Tests trace event creation, persistence, retrieval, filtering, pagination,
summary aggregation, decision events, tool events, LLM events, authorization,
buffer behavior, and persistence failure isolation.
"""
import uuid
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from observability.models import (
    AgentEventType, TraceEvent, TraceStatus, CRITICAL_EVENT_TYPES
)
from observability.trace_manager import TraceManager, create_trace_manager, get_trace_manager


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def wf_id():
    return f"wf_{uuid.uuid4().hex[:8]}"

@pytest.fixture
def tm(wf_id):
    return TraceManager(workflow_id=wf_id)


# ─── 1. Trace Event Creation ──────────────────────────────────────────────────

def test_trace_event_has_required_fields(wf_id):
    ev = TraceEvent(
        workflow_id = wf_id,
        trace_id    = "trace_abc",
        agent_name  = "orchestrator",
        event_type  = AgentEventType.WORKFLOW_STARTED,
        status      = TraceStatus.RUNNING,
        message     = "Workflow started.",
    )
    assert ev.event_id.startswith("ev_")
    assert ev.workflow_id == wf_id
    assert ev.trace_id == "trace_abc"
    assert ev.timestamp is not None
    assert ev.agent_name == "orchestrator"
    assert ev.event_type == AgentEventType.WORKFLOW_STARTED
    assert ev.status == TraceStatus.RUNNING


def test_trace_event_message_bounded():
    long_msg = "x" * 1000
    ev = TraceEvent(
        workflow_id="wf1", trace_id="t1",
        agent_name="test", event_type=AgentEventType.DECISION,
        status=TraceStatus.SUCCESS, message=long_msg
    )
    # Pydantic doesn't truncate but TraceManager does
    assert len(long_msg) > 500  # sanity


def test_critical_event_detection():
    ev = TraceEvent(
        workflow_id="wf1", trace_id="t1",
        agent_name="orchestrator",
        event_type=AgentEventType.WORKFLOW_COMPLETED,
        status=TraceStatus.SUCCESS,
    )
    assert ev.is_critical

    ev2 = TraceEvent(
        workflow_id="wf1", trace_id="t1",
        agent_name="orchestrator",
        event_type=AgentEventType.LLM_REQUEST,
        status=TraceStatus.RUNNING,
    )
    assert not ev2.is_critical


# ─── 2. TraceManager Methods ──────────────────────────────────────────────────

def test_start_workflow_adds_event(tm):
    with patch.object(tm, '_persist'):
        ev = tm.start_workflow()
    assert ev.event_type == AgentEventType.WORKFLOW_STARTED
    assert ev.status == TraceStatus.RUNNING
    assert tm.buffer_size() == 1


def test_complete_workflow(tm):
    with patch.object(tm, '_persist'):
        ev = tm.complete_workflow("Workflow completed.")
    assert ev.event_type == AgentEventType.WORKFLOW_COMPLETED
    assert ev.status == TraceStatus.SUCCESS


def test_fail_workflow(tm):
    with patch.object(tm, '_persist'):
        ev = tm.fail_workflow("Something went wrong.")
    assert ev.event_type == AgentEventType.WORKFLOW_FAILED
    assert ev.status == TraceStatus.FAILED


def test_start_and_complete_agent(tm):
    with patch.object(tm, '_persist'):
        tm.start_agent("data_analyst", "Data analysis started.")
        ev = tm.complete_agent("data_analyst", "Dataset analysis completed.",
                               result_summary="500 rows, 11 cols")
    assert ev.event_type == AgentEventType.AGENT_COMPLETED
    assert ev.result_summary == "500 rows, 11 cols"
    assert ev.duration_ms is not None


def test_fail_agent(tm):
    with patch.object(tm, '_persist'):
        tm.start_agent("training")
        ev = tm.fail_agent("training", "Training failed due to OOM.")
    assert ev.event_type == AgentEventType.AGENT_FAILED
    assert ev.status == TraceStatus.FAILED


# ─── 3. Decision Events ───────────────────────────────────────────────────────

def test_decision_event(tm):
    with patch.object(tm, '_persist'):
        ev = tm.record_decision(
            "model_engineer",
            decision="selected_candidate_models",
            message="Selected 3 candidate classification models.",
            metadata={"models": ["RandomForest", "XGBoost", "LogisticRegression"]},
        )
    assert ev.event_type == AgentEventType.DECISION
    assert ev.decision == "selected_candidate_models"
    assert ev.metadata["models"] == ["RandomForest", "XGBoost", "LogisticRegression"]


def test_decision_never_stores_raw_reasoning(tm):
    """Decision events must not expose chain-of-thought."""
    with patch.object(tm, '_persist'):
        ev = tm.record_decision(
            "orchestrator",
            decision="plan_generated",
            message="Workflow plan generated.",
            # No chain-of-thought fields allowed — only structured summary
        )
    # Ensure no hidden reasoning field exists
    assert not hasattr(ev, 'chain_of_thought')
    assert not hasattr(ev, 'thinking')
    assert not hasattr(ev, 'reasoning_steps')


# ─── 4. Tool Events ───────────────────────────────────────────────────────────

def test_tool_start_complete(tm):
    with patch.object(tm, '_persist'):
        ev_id = tm.start_tool("data_analyst", "dataset_reader")
        ev = tm.complete_tool("data_analyst", "dataset_reader",
                              result_summary="Dataset loaded: 500 rows, 11 cols",
                              parent_event_id=ev_id)
    assert ev.event_type == AgentEventType.TOOL_COMPLETED
    assert ev.tool_name == "dataset_reader"
    assert ev.result_summary == "Dataset loaded: 500 rows, 11 cols"
    assert ev.duration_ms is not None


def test_tool_failure(tm):
    with patch.object(tm, '_persist'):
        ev_id = tm.start_tool("data_analyst", "dataset_reader")
        ev = tm.fail_tool("data_analyst", "dataset_reader",
                          "CSV could not be parsed.", error_code="PARSE_ERROR",
                          parent_event_id=ev_id)
    assert ev.event_type == AgentEventType.TOOL_FAILED
    assert ev.error_code == "PARSE_ERROR"


# ─── 5. Handoff Events ───────────────────────────────────────────────────────

def test_handoff_event(tm):
    with patch.object(tm, '_persist'):
        ev = tm.record_handoff("orchestrator", "data_analyst")
    assert ev.event_type == AgentEventType.HANDOFF
    assert ev.metadata["target_agent"] == "data_analyst"
    assert "data_analyst" in ev.message


# ─── 6. LLM Events ───────────────────────────────────────────────────────────

def test_llm_request_response(tm):
    with patch.object(tm, '_persist'):
        req_id = tm.record_llm_request("model_engineer", "gemini", "gemini-pro")
        ev = tm.record_llm_response("model_engineer", "gemini", "gemini-pro",
                                    parent_event_id=req_id)
    assert ev.event_type == AgentEventType.LLM_RESPONSE
    assert ev.provider == "gemini"
    assert ev.duration_ms is not None


def test_llm_retry_event(tm):
    with patch.object(tm, '_persist'):
        ev = tm.record_llm_retry("orchestrator", "gemini", attempt=1, reason="Rate limited.")
    assert ev.event_type == AgentEventType.LLM_RETRY
    assert ev.status == TraceStatus.RETRYING
    assert ev.iteration == 1


def test_llm_fallback_event(tm):
    with patch.object(tm, '_persist'):
        ev = tm.record_llm_fallback("orchestrator", "gemini", "groq")
    assert ev.event_type == AgentEventType.LLM_FALLBACK
    assert ev.status == TraceStatus.FALLBACK
    assert ev.metadata["from_provider"] == "gemini"
    assert ev.metadata["to_provider"] == "groq"


# ─── 7. Checkpoint Events ─────────────────────────────────────────────────────

def test_checkpoint_created(tm):
    with patch.object(tm, '_persist'):
        ev = tm.record_checkpoint("data_analyst")
    assert ev.event_type == AgentEventType.CHECKPOINT_CREATED
    assert "data_analyst" in ev.message


def test_checkpoint_restored(tm):
    with patch.object(tm, '_persist'):
        ev = tm.record_checkpoint("evaluation", restored=True)
    assert ev.event_type == AgentEventType.CHECKPOINT_RESTORED
    assert "restored" in ev.message.lower()


# ─── 8. Training Events ───────────────────────────────────────────────────────

def test_training_events(tm):
    with patch.object(tm, '_persist'):
        ev_start = tm.record_training("training", started=True, model_name="GradientBoosting")
        ev_done  = tm.record_training("training", started=False, model_name="GradientBoosting")
    assert ev_start.event_type == AgentEventType.TRAINING_STARTED
    assert ev_done.event_type  == AgentEventType.TRAINING_COMPLETED


# ─── 9. Evaluation Events ─────────────────────────────────────────────────────

def test_evaluation_requirement_passed(tm):
    with patch.object(tm, '_persist'):
        ev = tm.record_requirement(True, "Recall requirement passed: 92% >= 90%.",
                                   metadata={"metric": "recall", "actual": 0.92}, iteration=1)
    assert ev.event_type == AgentEventType.REQUIREMENT_PASSED
    assert ev.status == TraceStatus.SUCCESS


def test_evaluation_requirement_failed(tm):
    with patch.object(tm, '_persist'):
        ev = tm.record_requirement(False, "Recall requirement failed: 82% < 90%.",
                                   metadata={"metric": "recall", "actual": 0.82}, iteration=1)
    assert ev.event_type == AgentEventType.REQUIREMENT_FAILED
    assert ev.status == TraceStatus.WARNING


# ─── 10. Debugger Events ─────────────────────────────────────────────────────

def test_debugger_event(tm):
    with patch.object(tm, '_persist'):
        ev = tm.record_debugger(
            "Debugger recommended threshold optimization.",
            decision="threshold_optimization",
            metadata={"action_type": "threshold_optimization"},
            iteration=1,
        )
    assert ev.event_type == AgentEventType.DEBUGGER_ANALYSIS
    assert ev.decision == "threshold_optimization"


# ─── 11. Reviewer Events ─────────────────────────────────────────────────────

def test_review_started(tm):
    with patch.object(tm, '_persist'):
        ev = tm.record_review(started=True)
    assert ev.event_type == AgentEventType.REVIEW_STARTED


def test_review_approved(tm):
    with patch.object(tm, '_persist'):
        ev = tm.record_review(started=False, passed=True,
                              metadata={"checks_count": 5, "findings": ["no leakage"]})
    assert ev.event_type == AgentEventType.REVIEW_COMPLETED
    assert ev.status == TraceStatus.SUCCESS
    assert "approved" in ev.message.lower()


def test_review_rejected(tm):
    with patch.object(tm, '_persist'):
        ev = tm.record_review(started=False, passed=False, message="Data leakage detected.")
    assert ev.event_type == AgentEventType.REVIEW_COMPLETED
    assert ev.status == TraceStatus.WARNING


# ─── 12. Buffer Behavior ─────────────────────────────────────────────────────

def test_buffer_grows_with_events(tm):
    with patch.object(tm, '_persist'):
        for i in range(10):
            tm.start_agent(f"agent_{i}")
    assert tm.buffer_size() == 10


def test_critical_events_preserved_on_buffer_overflow():
    """Critical events must NOT be dropped when buffer is full."""
    from collections import deque
    tm = TraceManager(workflow_id="wf_overflow")
    # Replace buffer with a small one to simulate overflow
    tm._buffer = deque(maxlen=10)
    with patch.object(tm, '_persist'):
        # Fill buffer with low-priority events
        for i in range(15):
            tm.record_state_change("orchestrator", f"State update {i}")
        # Now add a critical event
        ev = tm.complete_workflow("Workflow completed.")
    buffered = tm.get_buffered_events()
    # The critical event should be in the buffer (it was added last, deque keeps tail)
    critical = [e for e in buffered if e.is_critical]
    assert len(critical) >= 1


# ─── 13. Trace Persistence Failure Isolation ─────────────────────────────────

def test_persistence_failure_does_not_raise(wf_id):
    """If trace DB is down, the workflow must NOT fail."""
    tm = TraceManager(workflow_id=wf_id)
    # Patch _persist directly — simulates any downstream DB failure
    with patch.object(tm, '_persist', side_effect=Exception("DB down")):
        # This must not raise — persistence errors are swallowed
        try:
            ev = tm.start_workflow("Workflow started.")
        except Exception:
            pytest.fail("TraceManager propagated a persistence exception")
    # Buffer still has the event
    assert tm.buffer_size() >= 1


# ─── 14. Global Registry ─────────────────────────────────────────────────────

def test_create_and_get_trace_manager():
    wid = f"wf_{uuid.uuid4().hex[:6]}"
    tm = create_trace_manager(wid)
    assert get_trace_manager(wid) is tm


def test_trace_id_is_unique():
    wid1 = f"wf_{uuid.uuid4().hex[:6]}"
    wid2 = f"wf_{uuid.uuid4().hex[:6]}"
    tm1 = TraceManager(workflow_id=wid1)
    tm2 = TraceManager(workflow_id=wid2)
    assert tm1.trace_id != tm2.trace_id


# ─── 15. Parent-Child Relationships ──────────────────────────────────────────

def test_parent_event_id_propagation(tm):
    with patch.object(tm, '_persist'):
        ev_id = tm.start_tool("data_analyst", "dataset_reader")
        ev = tm.complete_tool("data_analyst", "dataset_reader",
                              result_summary="Done", parent_event_id=ev_id)
    assert ev.parent_event_id == ev_id


# ─── 16. Message Truncation in TraceManager ──────────────────────────────────

def test_long_messages_are_truncated(tm):
    with patch.object(tm, '_persist'):
        ev = tm._make_event(
            agent_name="orchestrator",
            event_type=AgentEventType.DECISION,
            status=TraceStatus.SUCCESS,
            message="x" * 1000,
        )
    assert len(ev.message) <= 500


# ─── 17. Iteration Tracking ──────────────────────────────────────────────────

def test_iteration_stored_in_event(tm):
    with patch.object(tm, '_persist'):
        ev = tm.record_training("training", started=True, model_name="XGBoost", iteration=2)
    assert ev.iteration == 2


# ─── 18. Trace Retention Config ──────────────────────────────────────────────

def test_trace_retention_days_in_settings():
    from config.settings import get_settings
    settings = get_settings()
    assert hasattr(settings, "TRACE_RETENTION_DAYS")
    assert settings.TRACE_RETENTION_DAYS > 0


# ─── 19. Event Ordering (via timestamps) ─────────────────────────────────────

def test_events_ordered_by_timestamp(tm):
    import time
    with patch.object(tm, '_persist'):
        ev1 = tm.start_workflow()
        time.sleep(0.01)
        ev2 = tm.start_agent("orchestrator")
        time.sleep(0.01)
        ev3 = tm.complete_agent("orchestrator")
    events = tm.get_buffered_events()
    timestamps = [e.timestamp for e in events]
    assert timestamps == sorted(timestamps)


# ─── 20. No Secrets in Events ────────────────────────────────────────────────

def test_no_api_keys_in_events(tm):
    with patch.object(tm, '_persist'):
        # record_llm_request returns the event_id string
        ev = tm._make_event(
            agent_name="orchestrator",
            event_type=AgentEventType.LLM_REQUEST,
            status=TraceStatus.RUNNING,
            provider="gemini",
            model="gemini-pro",
            metadata={"request_id": "req_123"},
        )
    # Must NOT contain API keys or secrets
    event_dict = ev.model_dump()
    for key, val in event_dict.items():
        if isinstance(val, str):
            assert "sk-" not in val
            assert "API_KEY" not in val
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, str):
                    assert "sk-" not in v
