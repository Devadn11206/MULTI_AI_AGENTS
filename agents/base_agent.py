from abc import ABC, abstractmethod
from typing import Optional, Any

class BaseAgent(ABC):
    """
    Abstract base class for all MLForge agents.
    """
    def __init__(self, llm_provider, name: str = "agent", role: str = "agent", system_prompt: str = "", state: Optional[Any] = None):
        self.llm_provider = llm_provider
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.state = state
        
    def run(self, user_input: str) -> str:
        try:
            resp = self.llm_provider.generate(prompt=user_input, system_prompt=self.system_prompt)
            return resp.content
        except Exception as e:
            return f"Agent {self.name} encountered an error: {e}"
