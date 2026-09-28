import pytest
from tools.review.leakage_checker import LeakageChecker
from tools.review.contamination_checker import ContaminationChecker
from tools.review.metric_validator import MetricValidator
from tools.review.overfitting_checker import OverfittingChecker
from tools.review.reproducibility_checker import ReproducibilityChecker
from tools.review.artifact_validator import ArtifactValidator
from tools.review.review_engine import ReviewEngine
from core.state.manager import AgentStateManager

class MockState:
    pass

def test_leakage_checker():
    state = MockState()
    state.experiment_plan = {"target_column": "churn"}
    checks = LeakageChecker.run_all(state)
    assert len(checks) == 3
    assert checks[1].name == "Target Leakage"
    assert checks[1].status == "PASS"

def test_contamination_checker():
    state = MockState()
    state.workflow_result = {"type": "dataset_profile", "data": {"duplicate_count": 5}}
    checks = ContaminationChecker.run_all(state)
    assert checks[1].name == "Duplicate Contamination"
    assert checks[1].status == "WARNING"

def test_metric_validator():
    state = MockState()
    state.experiment_plan = {"task_type": "classification", "primary_metric": "accuracy"}
    state.requirement_results = [{"status": "pass", "message": "Requirement satisfied."}]
    checks = MetricValidator.run_all(state)
    assert checks[0].status == "PASS"
    assert checks[2].status == "PASS"

def test_overfitting_checker():
    state = MockState()
    checks = OverfittingChecker.run_all(state)
    assert checks[0].status == "PASS"

def test_reproducibility_checker():
    state = MockState()
    state.experiment_plan = {"random_state": 42} # missing dataset_reference
    checks = ReproducibilityChecker.run_all(state)
    assert checks[0].status == "WARNING"

def test_artifact_validator():
    state = MockState()
    state.final_model_path = None
    checks = ArtifactValidator.run_all(state)
    assert checks[0].status == "FAIL"
    assert checks[0].severity == "CRITICAL"

def test_review_engine():
    engine = ReviewEngine()
    state = MockState()
    state.experiment_plan = {"task_type": "classification", "primary_metric": "accuracy", "random_state": 42, "dataset_reference": "dummy"}
    state.requirement_results = [{"status": "pass"}]
    state.final_model_path = "non_existent.pkl"
    
    res = engine.run_review(state)
    # Because final_model_path doesn't exist, artifact validator will fail critically
    assert res.status == "FAIL"
    assert len(res.critical_failures) > 0
    assert res.critical_failures[0]["name"] == "Model Existence"

def test_mock_llm_cannot_override_critical_fail():
    engine = ReviewEngine()
    state = MockState()
    # Mocking that all checks pass except one critical
    state.final_model_path = "invalid.pkl"
    res = engine.run_review(state)
    assert res.status == "FAIL"
    # Even if an LLM is asked to review `res`, the status of `res` is already deterministically "FAIL".
