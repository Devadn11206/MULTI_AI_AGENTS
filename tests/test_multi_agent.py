import pytest
from core.state.manager import AgentStateManager
from tools.registry import ToolRegistry
from tools.calculator import CalculatorTool
from agents.handoff import WorkerTask, WorkerResult, Handoff
from agents.worker_agent import WorkerAgent
from agents.manager_agent import ManagerAgent

class MockStructuredProvider:
    def __init__(self, sequence):
        self.sequence = sequence
        self.call_count = 0
    @property
    def provider_name(self) -> str: return "mock"
    def generate(self, prompt: str, system_prompt: str = None, response_schema: any = None):
        from core.llm.models import LLMResponse
        response = self.sequence[self.call_count]
        self.call_count += 1
        return LLMResponse(content=response, provider="mock", model="mock")

def test_handoff_and_worker():
    state = AgentStateManager()
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    
    worker_provider = MockStructuredProvider([
        '{"action": "tool_call", "tool_name": "calculator", "arguments": {"expression": "10 * 5"}}',
        '{"action": "task_completed", "result": "The answer is 50"}'
    ])
    
    worker = WorkerAgent(llm_provider=worker_provider, tool_registry=registry, state_manager=state)
    handoff = Handoff(state)
    
    res = handoff.delegate_to_worker(worker, "Calculate 10 * 5")
    
    assert res.status == "completed"
    assert res.result == "The answer is 50"
    
    # Verify state messages
    st = state.get_state()
    assert len(st.messages) == 2
    assert st.messages[0].metadata["sender"] == "manager"
    assert st.messages[1].metadata["sender"] == "worker"

def test_manager_direct_answer():
    state = AgentStateManager()
    manager_provider = MockStructuredProvider([
        '{"action": "answer_directly", "answer": "Hello from manager"}'
    ])
    
    manager = ManagerAgent(llm_provider=manager_provider, state_manager=state, worker_agent=None)
    ans = manager.run("Hi")
    assert ans == "Hello from manager"
    
    st = state.get_state()
    assert len(st.messages) == 2  # user + manager
    assert st.messages[-1].content == "Hello from manager"

def test_manager_delegation():
    state = AgentStateManager()
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    
    worker_provider = MockStructuredProvider([
        '{"action": "task_completed", "result": "Worker computed 10"}'
    ])
    worker = WorkerAgent(llm_provider=worker_provider, tool_registry=registry, state_manager=state)
    
    manager_provider = MockStructuredProvider([
        '{"action": "delegate", "task": "Compute this"}',
        '{"action": "answer_directly", "answer": "The final answer is 10"}'
    ])
    manager = ManagerAgent(llm_provider=manager_provider, state_manager=state, worker_agent=worker)
    
    ans = manager.run("What is 5*2?")
    assert ans == "The final answer is 10"
    
    st = state.get_state()
    assert len(st.decisions) >= 3 # user input, delegate, answer
    assert "The final answer is 10" in [m.content for m in st.messages]
