from core.llm_manager import LLMManager
from agents.base_agent import BaseAgent

class SimpleAgent(BaseAgent):
    def __init__(self, llm_provider: LLMManager):
        system_prompt = "You are a helpful AI assistant. Answer the user's questions clearly and accurately."
        super().__init__(name="SimpleAgent", llm_provider=llm_provider, system_prompt=system_prompt)
