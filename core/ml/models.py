from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class TrainingResult(BaseModel):
    experiment_id: str
    model_id: str
    model_name: str
    task_type: str
    status: str
    training_time_seconds: float = 0.0
    train_rows: int = 0
    test_rows: int = 0
    feature_count: int = 0
    artifact_path: Optional[str] = None
    preprocessing_pipeline_path: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None

class MetricResult(BaseModel):
    metric_name: str
    value: float
    dataset_split: str
    averaging: Optional[str] = None
    status: str = "completed"
    error: Optional[str] = None

class EvaluationResult(BaseModel):
    experiment_id: str
    model_id: str
    model_name: str
    task_type: str
    primary_metric: str
    primary_metric_value: Optional[float] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    dataset_split: str = "test"
    prediction_artifact: Optional[str] = None
    status: str
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None

class ModelComparisonResult(BaseModel):
    task_type: str
    primary_metric: str
    experiments: List[str] = Field(default_factory=list)
    ranking: List[Dict[str, Any]] = Field(default_factory=list)
    best_model: Optional[str] = None
    best_experiment_id: Optional[str] = None
    comparison_warnings: List[str] = Field(default_factory=list)
