from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class EvaluationResultBase(BaseModel):
    test_id: str
    test_category: Optional[str] = None
    dataset_id: Optional[str] = None
    status: str
    duration_ms: int = 0
    llm_calls: int = 0
    retries: int = 0
    fallbacks: int = 0
    best_model: Optional[str] = None
    best_metric: Optional[float] = None
    review_status: Optional[str] = None
    error_code: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None

class EvaluationResultCreate(EvaluationResultBase):
    evaluation_run_id: str

class EvaluationResultSchema(EvaluationResultBase):
    id: int
    evaluation_run_id: str
    
    class Config:
        from_attributes = True

class EvaluationRunBase(BaseModel):
    evaluation_run_id: str
    mode: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    duration_ms: int = 0
    summary: Optional[Dict[str, Any]] = None

class EvaluationRunCreate(EvaluationRunBase):
    pass

class EvaluationRunSchema(EvaluationRunBase):
    id: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: List[EvaluationResultSchema] = Field(default_factory=list)
    
    class Config:
        from_attributes = True
