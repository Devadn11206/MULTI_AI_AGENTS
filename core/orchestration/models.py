from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class MLProblem(BaseModel):
    problem_statement: str
    dataset_path: Optional[str] = None
    problem_type: Optional[str] = None
    target_column: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)
    optimization_metric: Optional[str] = None

class WorkflowStep(BaseModel):
    step_id: str
    name: str
    description: Optional[str] = None
    agent_role: str
    status: str = "pending" # pending, ready, running, completed, failed, skipped
    dependencies: List[str] = Field(default_factory=list)
    input_requirements: List[str] = Field(default_factory=list)
    output_requirements: List[str] = Field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0

class WorkflowPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    problem: Optional[MLProblem] = None
    steps: List[WorkflowStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "created" # created, running, completed, failed, paused
    current_step: Optional[str] = None
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)

class WorkflowResult(BaseModel):
    plan_id: str
    status: str
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    skipped_steps: List[str] = Field(default_factory=list)
    final_message: str
    errors: List[str] = Field(default_factory=list)
