"""
Phase 8 unit tests — Model Engineering Agent
All tests are deterministic. No real API keys required.
"""
import pytest
import json
from pydantic import ValidationError

from core.model_engineering.registry import get_registry, ModelMetadata
from core.model_engineering.factory import ModelFactory
from core.model_engineering.models import ExperimentPlan, ModelExperiment
from agents.model_engineering_agent import ModelEngineeringAgent
from core.state.manager import AgentStateManager
from core.analysis.models import DatasetProfile, ColumnProfile


# ────────────────────────────────────────────────────────────────────────────
# Mocks
# ────────────────────────────────────────────────────────────────────────────

class MockLLMManager:
    provider_name = "mock"
    
    def generate(self, prompt, system_prompt=None, response_schema=None):
        from core.llm.models import LLMResponse
        plan = {
            "experiment_id": "exp_test_123",
            "problem_description": "Predict churn",
            "task_type": "binary_classification",
            "target_column": "Churn",
            "dataset_reference": "data/processed/churn_features.csv",
            "candidate_models": [
                {
                    "model_id": "logistic_regression",
                    "model_name": "Logistic Regression",
                    "task_type": "binary_classification",
                    "priority": 1,
                    "rationale": "Baseline model"
                },
                {
                    "model_id": "random_forest_classifier",
                    "model_name": "Random Forest",
                    "task_type": "binary_classification",
                    "priority": 2,
                    "rationale": "Non-linear capabilities"
                }
            ],
            "baseline_model": "logistic_regression",
            "primary_metric": "recall",
            "secondary_metrics": ["f1", "precision"],
            "validation_strategy": "StratifiedKFold",
            "random_state": 42,
            "selection_reasoning": "Standard tabular approach"
        }
        return LLMResponse(content=json.dumps(plan), provider="mock", model="mock")


def get_mock_profile():
    from core.analysis.models import TargetDistribution
    return DatasetProfile(
        dataset_path="test.csv",
        file_type="csv",
        row_count=1000,
        column_count=5,
        columns=["Churn"],
        categorical_columns=["Churn"],
        target_distribution=TargetDistribution(
            target_column="Churn", 
            class_counts={"Yes": 500, "No": 500}, 
            class_percentages={"Yes": 50.0, "No": 50.0},
            number_of_classes=2, 
            imbalance_detected=False
        ),
        column_profiles={
            "Churn": ColumnProfile(name="Churn", dtype="object", non_null_count=1000, unique_count=2, missing_count=0, missing_percentage=0.0)
        }
    )

def get_mock_fe_result():
    from core.feature_engineering.models import FeatureEngineeringResult
    return FeatureEngineeringResult(
        status="completed",
        original_dataset_path="test.csv",
        processed_dataset_path="data/processed/test_features.csv",
        target_column="Churn",
        original_shape=(1000, 5),
        transformed_shape=(1000, 8),
        missing_before=0,
        missing_after=0,
        duplicates_before=0,
        duplicates_after=0
    )


# ────────────────────────────────────────────────────────────────────────────
# Registry & Factory Tests
# ────────────────────────────────────────────────────────────────────────────

def test_registry_list_classification_models():
    reg = get_registry()
    models = reg.list_models(task_type="binary_classification")
    ids = [m.model_id for m in models]
    assert "logistic_regression" in ids
    assert "random_forest_classifier" in ids
    assert "linear_regression" not in ids

def test_registry_list_regression_models():
    reg = get_registry()
    models = reg.list_models(task_type="regression")
    ids = [m.model_id for m in models]
    assert "linear_regression" in ids
    assert "random_forest_regressor" in ids
    assert "logistic_regression" not in ids

def test_registry_get_metadata():
    reg = get_registry()
    meta = reg.get_metadata("logistic_regression")
    assert meta is not None
    assert meta.model_family == "linear"
    assert meta.requires_scaling is True

def test_factory_valid_model():
    res = ModelFactory.create("logistic_regression", random_state=123)
    assert type(res).__name__ == "LogisticRegression"
    assert res.random_state == 123

def test_factory_invalid_model():
    with pytest.raises(ValueError, match="Unknown model_id"):
        ModelFactory.create("non_existent_model")


# ────────────────────────────────────────────────────────────────────────────
# Agent Tests
# ────────────────────────────────────────────────────────────────────────────

class TestModelEngineeringAgent:
    def setup_method(self):
        self.state_manager = AgentStateManager()
        self.agent = ModelEngineeringAgent(MockLLMManager(), self.state_manager)
        
    def test_task_detection_binary(self):
        profile = get_mock_profile()
        task = self.agent._determine_task_type("Churn", profile)
        assert task == "binary_classification"
        
    def test_task_detection_multiclass(self):
        profile = DatasetProfile(
            dataset_path="test.csv", file_type="csv",
            row_count=100, column_count=2, columns=["Target"],
            categorical_columns=["Target"],
            column_profiles={"Target": ColumnProfile(name="Target", dtype="object", non_null_count=100, unique_count=5, missing_count=0, missing_percentage=0.0)}
        )
        from core.analysis.models import TargetDistribution
        profile.target_distribution = TargetDistribution(
            target_column="Target", 
            class_counts={"1": 20, "2": 20, "3": 20, "4": 20, "5": 20}, 
            class_percentages={"1": 20.0, "2": 20.0, "3": 20.0, "4": 20.0, "5": 20.0},
            number_of_classes=5, 
            imbalance_detected=False
        )
        task = self.agent._determine_task_type("Target", profile)
        assert task == "multiclass_classification"
        
    def test_task_detection_regression(self):
        profile = DatasetProfile(
            dataset_path="test.csv", file_type="csv",
            row_count=100, column_count=2, columns=["Target"],
            numeric_columns=["Target"],
            column_profiles={"Target": ColumnProfile(name="Target", dtype="float64", non_null_count=100, unique_count=50, missing_count=0, missing_percentage=0.0)}
        )
        task = self.agent._determine_task_type("Target", profile)
        assert task == "regression"

    def test_plan_experiment_returns_valid_plan(self):
        plan = self.agent.plan_experiment(
            ml_problem="Predict churn",
            target_column="Churn",
            dataset_profile=get_mock_profile(),
            feature_engineering_result=get_mock_fe_result()
        )
        assert isinstance(plan, ExperimentPlan)
        assert plan.task_type == "binary_classification"
        assert len(plan.candidate_models) == 2
        assert plan.baseline_model == "logistic_regression"
        
    def test_agent_state_updated(self):
        self.agent.plan_experiment(
            ml_problem="Predict churn",
            target_column="Churn",
            dataset_profile=get_mock_profile(),
            feature_engineering_result=get_mock_fe_result()
        )
        st = self.state_manager.get_state()
        assert st.experiment_plan is not None
        assert st.selected_models == ["logistic_regression", "random_forest_classifier"]

    def test_validate_and_filter_plan_removes_uninstalled(self):
        plan = ExperimentPlan(
            experiment_id="1", problem_description="1", task_type="binary_classification",
            target_column="1", dataset_reference="1", baseline_model="1", primary_metric="1",
            validation_strategy="1", random_state=1, selection_reasoning="1",
            candidate_models=[
                ModelExperiment(model_id="logistic_regression", model_name="LR", task_type="bc", priority=1, rationale=""),
                ModelExperiment(model_id="hallucinated_model", model_name="HM", task_type="bc", priority=2, rationale="")
            ]
        )
        installed = get_registry().list_models("binary_classification")
        self.agent._validate_and_filter_plan(plan, installed)
        assert len(plan.candidate_models) == 1
        assert plan.candidate_models[0].model_id == "logistic_regression"
        assert len(plan.warnings) == 1
        assert "hallucinated_model" in plan.warnings[0]
