from typing import Optional, Dict, Any, List
from pydantic import BaseModel

class WorkflowResultResponse(BaseModel):
    workflow_id: str
    objective: str
    status: str
    dataset_info: Optional[Dict[str, Any]] = None
    task_type: Optional[str] = None
    dataset_profile: Optional[Dict[str, Any]] = None
    feature_engineering: Optional[Dict[str, Any]] = None
    experiment_plan: Optional[Dict[str, Any]] = None
    training_results: List[Dict[str, Any]] = []
    evaluation_results: List[Dict[str, Any]] = []
    self_correction_history: List[Dict[str, Any]] = []
    reviewer_result: Optional[Dict[str, Any]] = None
    best_model: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = []
    warnings: List[str] = []
