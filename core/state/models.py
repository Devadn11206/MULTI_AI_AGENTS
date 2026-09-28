import uuid
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from datetime import datetime

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

class Decision(BaseModel):
    action: str
    tool_name: Optional[str] = None
    reason: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Plan(BaseModel):
    goal: str
    steps: List[str]
    current_step: int = 0
    status: str = "pending"

class DatasetInfo(BaseModel):
    name: Optional[str] = None
    path: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    target_column: Optional[str] = None

class Experiment(BaseModel):
    experiment_id: str
    model_name: str
    parameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    status: str = "pending"

class ModelInfo(BaseModel):
    name: str
    type: str
    path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None

class EvaluationResult(BaseModel):
    metrics: Dict[str, float]
    passed: bool
    issues: List[str] = Field(default_factory=list)

class ErrorInfo(BaseModel):
    source: str
    message: str
    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AgentState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: Optional[str] = None
    user_task: Optional[str] = None
    current_plan: Optional[Plan] = None
    messages: List[Message] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    tool_results: List[Any] = Field(default_factory=list) 
    dataset_info: Optional[DatasetInfo] = None
    experiments: List[Experiment] = Field(default_factory=list)
    best_model: Optional[ModelInfo] = None
    evaluation: Optional[EvaluationResult] = None
    errors: List[ErrorInfo] = Field(default_factory=list)
    
    # Phase 5: Orchestration fields
    ml_problem: Optional[Dict[str, Any]] = None
    workflow_plan: Optional[Dict[str, Any]] = None
    current_workflow_step: Optional[str] = None
    workflow_result: Optional[Dict[str, Any]] = None

    # Phase 7: Feature Engineering fields
    feature_engineering_plan: Optional[Dict[str, Any]] = None
    feature_engineering_result: Optional[Dict[str, Any]] = None
    processed_dataset_reference: Optional[str] = None

    # Phase 8: Model Engineering fields
    experiment_plan: Optional[Dict[str, Any]] = None
    model_candidates: Optional[List[Dict[str, Any]]] = None
    selected_models: Optional[List[str]] = None
    model_engineering_warnings: Optional[List[str]] = None

    # Phase 9: Training & Evaluation
    test_dataset_path: Optional[str] = None
    training_results: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_results: List[Dict[str, Any]] = Field(default_factory=list)
    model_comparison: Optional[Dict[str, Any]] = None
    best_model_id: Optional[str] = None

    # Phase 10: Debugger & Self-Correction
    requirements: Optional[List[Dict[str, Any]]] = None
    requirement_results: Optional[List[Dict[str, Any]]] = None
    current_iteration: int = 1
    max_iterations: int = 3
    attempt_history: List[Dict[str, Any]] = Field(default_factory=list)
    debug_history: List[Dict[str, Any]] = Field(default_factory=list)
    correction_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_debug_recommendation: Optional[Dict[str, Any]] = None
    self_correction_status: str = "pending"
    best_attempt: Optional[int] = None
    final_model_reference: Optional[str] = None
    
    # Phase 11: Reviewer Agent
    review_results: List[Dict[str, Any]] = Field(default_factory=list)
    review_history: List[Dict[str, Any]] = Field(default_factory=list)
    quality_gate_status: Optional[str] = None
    critical_failures: List[Dict[str, Any]] = Field(default_factory=list)
    review_warnings: List[Dict[str, Any]] = Field(default_factory=list)
    review_recommendations: List[str] = Field(default_factory=list)
    final_review: Optional[Dict[str, Any]] = None
