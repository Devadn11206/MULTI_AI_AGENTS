import pytest
from core.state.manager import AgentStateManager
from core.state.models import Plan, DatasetInfo, Experiment, ModelInfo, EvaluationResult
from tools.base_tool import ToolResult

def test_state_creation_and_fields():
    manager = AgentStateManager()
    state = manager.get_state()
    assert state.session_id is not None
    assert len(state.messages) == 0
    assert len(state.decisions) == 0

def test_state_updates():
    manager = AgentStateManager()
    manager.set_user_task("Test task")
    manager.add_message("user", "Hello")
    manager.add_decision("tool_call", tool_name="calculator")
    manager.add_tool_result(ToolResult(success=True, tool_name="calculator", result="100"))
    manager.set_plan(Plan(goal="Test plan", steps=["Step 1"]))
    manager.set_dataset_info(DatasetInfo(name="test_data"))
    manager.add_experiment(Experiment(experiment_id="exp1", model_name="model1"))
    manager.set_best_model(ModelInfo(name="model1", type="regression"))
    manager.set_evaluation(EvaluationResult(metrics={"acc": 0.9}, passed=True))
    manager.add_error("Test", "Test error", "ErrorType")
    
    state = manager.get_state()
    assert state.user_task == "Test task"
    assert len(state.messages) == 1
    assert len(state.decisions) == 1
    assert len(state.tool_results) == 1
    assert state.current_plan.goal == "Test plan"
    assert state.dataset_info.name == "test_data"
    assert len(state.experiments) == 1
    assert state.best_model.name == "model1"
    assert state.evaluation.passed is True
    assert len(state.errors) == 1

def test_state_persistence(tmp_path):
    manager = AgentStateManager()
    manager.set_user_task("Persistence test")
    
    save_path = str(tmp_path / "state.json")
    manager.save_state(save_path)
    
    new_manager = AgentStateManager()
    new_manager.load_state(save_path)
    
    assert new_manager.get_state().user_task == "Persistence test"
    assert new_manager.get_state().session_id == manager.get_state().session_id

def test_build_context():
    manager = AgentStateManager()
    manager.set_user_task("Context test")
    manager.add_message("user", "Hello Context")
    manager.add_message("assistant", "Hi there")
    
    context = manager.build_context()
    assert "Context test" in context
    assert "[user] Hello Context" in context
    assert "[assistant] Hi there" in context
