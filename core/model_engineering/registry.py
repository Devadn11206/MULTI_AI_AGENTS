"""
Phase 8: Centralized Model Registry
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    model_id: str
    display_name: str
    task_types: List[str]
    model_family: str
    supports_probability: bool
    supports_multiclass: bool
    requires_scaling: bool
    handles_non_linear_relationships: bool
    handles_missing_values: bool
    interpretability: str
    computational_cost: str
    typical_strengths: str
    typical_weaknesses: str
    dependencies: List[str] = Field(default_factory=list)


class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, ModelMetadata] = {}
        self._register_defaults()

    def _register_defaults(self):
        # CLASSIFICATION
        self.register(ModelMetadata(
            model_id="logistic_regression",
            display_name="Logistic Regression",
            task_types=["binary_classification", "multiclass_classification"],
            model_family="linear",
            supports_probability=True,
            supports_multiclass=True,
            requires_scaling=True,
            handles_non_linear_relationships=False,
            handles_missing_values=False,
            interpretability="high",
            computational_cost="low",
            typical_strengths="Provides a simple interpretable baseline. Fast training.",
            typical_weaknesses="Struggles with non-linear relationships.",
            dependencies=["sklearn"]
        ))
        
        self.register(ModelMetadata(
            model_id="random_forest_classifier",
            display_name="Random Forest Classifier",
            task_types=["binary_classification", "multiclass_classification"],
            model_family="tree_ensemble",
            supports_probability=True,
            supports_multiclass=True,
            requires_scaling=False,
            handles_non_linear_relationships=True,
            handles_missing_values=False,
            interpretability="medium",
            computational_cost="medium",
            typical_strengths="Handles nonlinear relationships and mixed feature interactions well. Robust to outliers.",
            typical_weaknesses="Can overfit noisy datasets. Large memory footprint.",
            dependencies=["sklearn"]
        ))

        self.register(ModelMetadata(
            model_id="gradient_boosting_classifier",
            display_name="Gradient Boosting Classifier",
            task_types=["binary_classification", "multiclass_classification"],
            model_family="tree_ensemble",
            supports_probability=True,
            supports_multiclass=True,
            requires_scaling=False,
            handles_non_linear_relationships=True,
            handles_missing_values=False,
            interpretability="low",
            computational_cost="high",
            typical_strengths="Often yields high accuracy on tabular data.",
            typical_weaknesses="Slower training time. Sensitive to hyperparameters.",
            dependencies=["sklearn"]
        ))
        
        self.register(ModelMetadata(
            model_id="xgboost_classifier",
            display_name="XGBoost Classifier",
            task_types=["binary_classification", "multiclass_classification"],
            model_family="tree_ensemble",
            supports_probability=True,
            supports_multiclass=True,
            requires_scaling=False,
            handles_non_linear_relationships=True,
            handles_missing_values=True,
            interpretability="low",
            computational_cost="high",
            typical_strengths="Advanced gradient boosting. Very fast and highly performant.",
            typical_weaknesses="Requires parameter tuning. External dependency.",
            dependencies=["xgboost"]
        ))

        self.register(ModelMetadata(
            model_id="svm_classifier",
            display_name="Support Vector Machine (Classifier)",
            task_types=["binary_classification", "multiclass_classification"],
            model_family="svm",
            supports_probability=True,
            supports_multiclass=True,
            requires_scaling=True,
            handles_non_linear_relationships=True,
            handles_missing_values=False,
            interpretability="low",
            computational_cost="high",
            typical_strengths="Effective in high-dimensional spaces.",
            typical_weaknesses="Does not scale well to very large datasets.",
            dependencies=["sklearn"]
        ))

        # REGRESSION
        self.register(ModelMetadata(
            model_id="linear_regression",
            display_name="Linear Regression",
            task_types=["regression"],
            model_family="linear",
            supports_probability=False,
            supports_multiclass=False,
            requires_scaling=True,
            handles_non_linear_relationships=False,
            handles_missing_values=False,
            interpretability="high",
            computational_cost="low",
            typical_strengths="Provides a simple interpretable baseline.",
            typical_weaknesses="Assumes linear relationship.",
            dependencies=["sklearn"]
        ))
        
        self.register(ModelMetadata(
            model_id="random_forest_regressor",
            display_name="Random Forest Regressor",
            task_types=["regression"],
            model_family="tree_ensemble",
            supports_probability=False,
            supports_multiclass=False,
            requires_scaling=False,
            handles_non_linear_relationships=True,
            handles_missing_values=False,
            interpretability="medium",
            computational_cost="medium",
            typical_strengths="Handles non-linear patterns robustly.",
            typical_weaknesses="Cannot extrapolate outside training bounds.",
            dependencies=["sklearn"]
        ))
        
        self.register(ModelMetadata(
            model_id="gradient_boosting_regressor",
            display_name="Gradient Boosting Regressor",
            task_types=["regression"],
            model_family="tree_ensemble",
            supports_probability=False,
            supports_multiclass=False,
            requires_scaling=False,
            handles_non_linear_relationships=True,
            handles_missing_values=False,
            interpretability="low",
            computational_cost="high",
            typical_strengths="High accuracy for tabular regression.",
            typical_weaknesses="Slower training time.",
            dependencies=["sklearn"]
        ))

        self.register(ModelMetadata(
            model_id="xgboost_regressor",
            display_name="XGBoost Regressor",
            task_types=["regression"],
            model_family="tree_ensemble",
            supports_probability=False,
            supports_multiclass=False,
            requires_scaling=False,
            handles_non_linear_relationships=True,
            handles_missing_values=True,
            interpretability="low",
            computational_cost="high",
            typical_strengths="Fast and accurate advanced boosting.",
            typical_weaknesses="External dependency.",
            dependencies=["xgboost"]
        ))

        self.register(ModelMetadata(
            model_id="svm_regressor",
            display_name="Support Vector Regressor",
            task_types=["regression"],
            model_family="svm",
            supports_probability=False,
            supports_multiclass=False,
            requires_scaling=True,
            handles_non_linear_relationships=True,
            handles_missing_values=False,
            interpretability="low",
            computational_cost="high",
            typical_strengths="Effective for high-dimensional complex datasets.",
            typical_weaknesses="Poor scalability on large data.",
            dependencies=["sklearn"]
        ))

    def register(self, metadata: ModelMetadata):
        self._models[metadata.model_id] = metadata

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        return self._models.get(model_id)

    def list_models(self, task_type: Optional[str] = None) -> List[ModelMetadata]:
        if not task_type:
            return list(self._models.values())
        return [m for m in self._models.values() if task_type in m.task_types]

    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        return self.get_model(model_id)

    def is_available(self, model_id: str) -> bool:
        meta = self.get_model(model_id)
        if not meta:
            return False
        
        for dep in meta.dependencies:
            try:
                __import__(dep)
            except ImportError:
                return False
        return True

_registry = ModelRegistry()

def get_registry() -> ModelRegistry:
    return _registry
