import pytest
from typing import Optional, Type
from pydantic import BaseModel
from core.llm_manager import LLMManager
from tools.registry import ToolRegistry
from tools.calculator import CalculatorTool
from agents.tool_agent import ToolUsingAgent, ToolCallDecision
from core.state.manager import AgentStateManager

class MockStructuredProvider(LLMManager):
    def __init__(self, sequence):
        self.sequence = sequence
        self.call_count = 0
    @property
    def provider_name(self) -> str: return "mock"
    def generate(self, prompt: str, system_prompt: Optional[str] = None, response_schema: Optional[Type[BaseModel]] = None):
        from core.llm.models import LLMResponse
        response = self.sequence[self.call_count]
        self.call_count += 1
        return LLMResponse(content=response, provider="mock", model="mock")

def test_tool_agent_final_answer():
    provider = MockStructuredProvider([
        '{"action": "final_answer", "answer": "This is a direct answer."}'
    ])
    registry = ToolRegistry()
    state_manager = AgentStateManager()
    agent = ToolUsingAgent(provider, registry, state_manager)
    
    result = agent.run("Hello")
    assert result == "This is a direct answer."
    
    state = state_manager.get_state()
    assert len(state.messages) == 2 # user and assistant
    assert state.messages[-1].content == "This is a direct answer."

def test_tool_agent_tool_call():
    provider = MockStructuredProvider([
        '{"action": "tool_call", "tool_name": "calculator", "arguments": {"expression": "10 * 5"}}',
        '{"action": "final_answer", "answer": "The answer is 50."}'
    ])
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    state_manager = AgentStateManager()
    agent = ToolUsingAgent(provider, registry, state_manager)
    
    result = agent.run("What is 10 * 5?")
    assert result == "The answer is 50."
    
    state = state_manager.get_state()
    assert len(state.decisions) == 1
    assert len(state.tool_results) == 1
    assert state.tool_results[0].result == "50"

def test_tool_agent_invalid_tool():
    provider = MockStructuredProvider([
        '{"action": "tool_call", "tool_name": "unknown", "arguments": {}}',
        '{"action": "final_answer", "answer": "I failed because tool is unknown."}'
    ])
    registry = ToolRegistry()
    state_manager = AgentStateManager()
    agent = ToolUsingAgent(provider, registry, state_manager)
    
    result = agent.run("Do something")
    assert result == "I failed because tool is unknown."
    
    state = state_manager.get_state()
    assert len(state.errors) > 0
