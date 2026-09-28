from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class ErrorResponse(BaseModel):
    code: str
    message: str
    workflow_id: Optional[str] = None
    
class SuccessResponse(BaseModel):
    status: str = "success"
    message: str
