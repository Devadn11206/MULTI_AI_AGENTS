import uuid
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    fallback_used: bool = False
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    latency_ms: int = 0

class ProviderState(BaseModel):
    provider_name: str
    status: str = "AVAILABLE"  # AVAILABLE, DEGRADED, RATE_LIMITED, UNAVAILABLE, CONFIGURATION_ERROR
    last_error: Optional[str] = None
    last_success: float = 0.0
    failure_count: int = 0
    success_count: int = 0
    cooldown_until: float = 0.0
    total_requests: int = 0
    total_tokens: int = 0

class BaseProvider(ABC):
    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model
        self.state = ProviderState(provider_name=name)

    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None, temperature: Optional[float] = None, response_schema: Optional[Any] = None) -> LLMResponse:
        pass
