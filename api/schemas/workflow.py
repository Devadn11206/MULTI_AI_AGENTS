from typing import Optional, Dict, Any, List
from pydantic import BaseModel, field_validator

class WorkflowConstraints(BaseModel):
    max_iterations: int = 3
    time_limit_seconds: Optional[int] = 600

class WorkflowCreateRequest(BaseModel):
    dataset_id: str
    objective: str
    priority: Optional[str] = "recall"
    target_column: Optional[str] = None
    task_type: Optional[str] = "auto"
    constraints: Optional[WorkflowConstraints] = WorkflowConstraints()

    @field_validator('objective')
    def objective_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Objective must not be empty')
        return v
        
    @field_validator('task_type')
    def validate_task_type(cls, v):
        allowed = ["classification", "regression", "auto"]
        if v not in allowed:
            raise ValueError(f"Task type must be one of {allowed}")
        return v

class WorkflowCreateResponse(BaseModel):
    workflow_id: str
    status: str

class WorkflowStatusResponse(BaseModel):
    workflow_id: str
    status: str
    current_agent: Optional[str] = None
    progress: int = 0
    message: Optional[str] = None

class WorkflowEvent(BaseModel):
    event_id: str
    workflow_id: str
    timestamp: str
    agent: str
    event_type: str
    status: str
    message: str
    metadata: Optional[Dict[str, Any]] = None

class WorkflowEventsResponse(BaseModel):
    events: List[WorkflowEvent]
