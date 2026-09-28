from abc import ABC, abstractmethod
from typing import Optional, Type
from pydantic import BaseModel
from core.llm.models import LLMResponse

class LLMManager(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass
