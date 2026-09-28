from typing import Optional, Dict, Any
from pydantic import BaseModel

class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
