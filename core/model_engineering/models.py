"""
Phase 8: Model Engineering Data Models
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ModelExperiment(BaseModel):
    model_id: str
    model_name: str
    task_type: str
    hyperparameter_strategy: str = Field(default="baseline_defaults")
    preprocessing_requirements: List[str] = Field(default_factory=list)
    evaluation_metrics: List[str] = Field(default_factory=list)
    expected_characteristics: str = Field(default="")
    priority: int
    rationale: str


class ExperimentConstraints(BaseModel):
    priority_metric: Optional[str] = None
    max_training_time_minutes: Optional[int] = None
    avoid_complex_models: Optional[bool] = None
    require_interpretability: Optional[bool] = None


class ExperimentPlan(BaseModel):
    experiment_id: str
    problem_description: str
    task_type: str
    target_column: str
    dataset_reference: str
    candidate_models: List[ModelExperiment] = Field(default_factory=list)
    baseline_model: str
    primary_metric: str
    secondary_metrics: List[str] = Field(default_factory=list)
    validation_strategy: str
    preprocessing_requirements: List[str] = Field(default_factory=list)
    class_imbalance_strategy: Optional[str] = None
    random_state: int
    constraints: Optional[Dict[str, Any]] = None
    selection_reasoning: str
    warnings: List[str] = Field(default_factory=list)
