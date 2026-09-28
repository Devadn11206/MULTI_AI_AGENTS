import pytest
from agents.base_agent import BaseAgent
from agents.simple_agent import SimpleAgent
from core.llm_manager import LLMManager

from core.llm.models import LLMResponse

class MockProvider(LLMManager):
    @property
    def provider_name(self) -> str: return "mock"
    def generate(self, prompt: str, system_prompt: str | None = None, response_schema=None) -> LLMResponse:
        if system_prompt:
            return LLMResponse(content=f"SYSTEM: {system_prompt} | PROMPT: {prompt}", provider="mock", model="mock")
        return LLMResponse(content=f"PROMPT: {prompt}", provider="mock", model="mock")

class FailingProvider(LLMManager):
    @property
    def provider_name(self) -> str: return "fail"
    def generate(self, prompt: str, system_prompt: str | None = None, response_schema=None) -> LLMResponse:
        raise RuntimeError("API failure")

def test_base_agent():
    provider = MockProvider()
    agent = BaseAgent(name="TestAgent", llm_provider=provider, system_prompt="Test System")
    response = agent.run("Hello")
    assert response == "SYSTEM: Test System | PROMPT: Hello"

def test_base_agent_error_handling():
    provider = FailingProvider()
    agent = BaseAgent(name="TestAgent", llm_provider=provider)
    response = agent.run("Hello")
    assert "Agent TestAgent encountered an error: API failure" in response

def test_simple_agent():
    provider = MockProvider()
    agent = SimpleAgent(llm_provider=provider)
    response = agent.run("Hello")
    assert "SYSTEM: You are a helpful AI assistant" in response
    assert "PROMPT: Hello" in response
