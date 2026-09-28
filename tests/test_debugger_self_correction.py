import pytest
from debug.action_registry import ActionRegistry
from debug.requirement_evaluator import RequirementEvaluator, MetricRequirement

def test_requirement_evaluator():
    metrics = {"recall": 0.85, "precision": 0.70, "rmse": 12.5}
    
    # Test >=
    req_recall_pass = MetricRequirement("recall", ">=", 0.80)
    res = RequirementEvaluator.evaluate(metrics, req_recall_pass)
    assert res.status == "pass"
    
    req_recall_fail = MetricRequirement("recall", ">=", 0.90)
    res = RequirementEvaluator.evaluate(metrics, req_recall_fail)
    assert res.status == "fail"
    
    # Test <=
    req_rmse_pass = MetricRequirement("rmse", "<=", 15.0)
    res = RequirementEvaluator.evaluate(metrics, req_rmse_pass)
    assert res.status == "pass"
    
    req_rmse_fail = MetricRequirement("rmse", "<=", 10.0)
    res = RequirementEvaluator.evaluate(metrics, req_rmse_fail)
    assert res.status == "fail"
    
    # Test ==
    req_eq = MetricRequirement("recall", "==", 0.85)
    res = RequirementEvaluator.evaluate(metrics, req_eq)
    assert res.status == "pass"

def test_action_registry():
    registry = ActionRegistry()
    
    # Valid action
    assert registry.is_valid_action("class_weight", "classification", "logistic_regression") is True
    
    # Invalid action ID
    assert registry.is_valid_action("fake_action", "classification", "logistic_regression") is False
    
    # Incompatible task type
    assert registry.is_valid_action("class_weight", "regression", "logistic_regression") is False
    
    # Unsupported model
    assert registry.is_valid_action("class_weight", "classification", "some_unsupported_model") is False
    
    # Universal support
    assert registry.is_valid_action("model_change", "regression", "any_model") is True
