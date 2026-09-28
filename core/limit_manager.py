from typing import Dict, Any
from config.settings import get_settings
from core.errors import AgentCallLimitExceeded

class LimitManager:
    def __init__(self):
        settings = get_settings()
        self.max_total_calls = settings.MAX_AGENT_CALLS
        self.max_agent_calls = settings.MAX_CALLS_PER_AGENT
        
        self.total_calls = 0
        self.agent_calls: Dict[str, int] = {}

    def check_and_increment(self, agent_name: str):
        if self.total_calls >= self.max_total_calls:
            raise AgentCallLimitExceeded(f"Maximum total agent calls ({self.max_total_calls}) reached.")
            
        current = self.agent_calls.get(agent_name, 0)
        if current >= self.max_agent_calls:
            raise AgentCallLimitExceeded(f"Maximum calls ({self.max_agent_calls}) for agent '{agent_name}' reached.")
            
        self.total_calls += 1
        self.agent_calls[agent_name] = current + 1

    def get_usage(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "max_total_calls": self.max_total_calls,
            "agent_calls": self.agent_calls
        }
