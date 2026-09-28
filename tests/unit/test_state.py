import pytest
from core.state.manager import AgentStateManager

def test_state_manager_initialization():
    sm = AgentStateManager("test_wf_123")
    state = sm.get_state()
    assert state.workflow_id == "test_wf_123"
    assert state.current_phase == "planning"

def test_state_manager_history():
    sm = AgentStateManager("test_wf_123")
    sm.state.attempt_history.append({"status": "pass"})
    assert len(sm.state.attempt_history) == 1
    assert sm.state.attempt_history[0]["status"] == "pass"
