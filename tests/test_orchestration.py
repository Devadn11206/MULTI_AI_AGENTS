import pytest
from core.orchestration.models import MLProblem, WorkflowPlan, WorkflowStep
from core.orchestration.planner import Planner
from core.orchestration.router import Router
from core.orchestration.executor import Executor
from core.orchestration.orchestrator import Orchestrator
from core.state.manager import AgentStateManager
from agents.worker_agent import WorkerAgent
from tools.registry import ToolRegistry

class MockPlannerProvider:
    @property
    def provider_name(self) -> str: return "mock"
    def generate(self, prompt, system_prompt, response_schema):
        from core.llm.models import LLMResponse
        return LLMResponse(content='{"goal": "Test Goal", "steps": [{"step_id": "01", "name": "Test Step", "agent_role": "data_analyst", "dependencies": []}]}', provider="mock", model="mock")

class MockWorkerAgent:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
    def execute_task(self, task):
        class MockRes:
            def __init__(self, f):
                self.status = "failed" if f else "completed"
                self.result = "Success"
                self.errors = ["Error"] if f else []
        return MockRes(self.should_fail)

def test_ml_problem():
    p = MLProblem(problem_statement="Test")
    assert p.problem_statement == "Test"
    assert p.constraints == []

def test_planner():
    planner = Planner(MockPlannerProvider())
    plan = planner.generate_plan(MLProblem(problem_statement="Test"))
    assert plan.goal == "Test Goal"
    assert len(plan.steps) == 1
    planner.validate_plan(plan) # should pass

def test_planner_validation_circular():
    planner = Planner(MockPlannerProvider())
    plan = WorkflowPlan(goal="Goal", steps=[
        WorkflowStep(step_id="1", name="A", agent_role="A", dependencies=["2"]),
        WorkflowStep(step_id="2", name="B", agent_role="B", dependencies=["1"])
    ])
    with pytest.raises(ValueError, match="Circular dependency"):
        planner.validate_plan(plan)

def test_router():
    router = Router()
    plan = WorkflowPlan(goal="Goal", steps=[
        WorkflowStep(step_id="1", name="A", agent_role="A", dependencies=[]),
        WorkflowStep(step_id="2", name="B", agent_role="B", dependencies=["1"])
    ])
    
    s1 = router.get_next_step(plan)
    assert s1.step_id == "1"
    
    s1.status = "running"
    assert router.get_next_step(plan) is None
    
    s1.status = "completed"
    plan.completed_steps.append("1")
    
    s2 = router.get_next_step(plan)
    assert s2.step_id == "2"
    
    s2.status = "failed"
    plan.failed_steps.append("2")
    
    assert router.get_next_step(plan) is None

def test_executor():
    sm = AgentStateManager()
    ex = Executor(sm, MockWorkerAgent())
    
    plan = WorkflowPlan(goal="Goal", steps=[
        WorkflowStep(step_id="1", name="A", agent_role="A", dependencies=[])
    ])
    
    res = ex.execute_plan(plan)
    assert res.status == "completed"
    assert "1" in res.completed_steps

def test_executor_fail():
    sm = AgentStateManager()
    ex = Executor(sm, MockWorkerAgent(should_fail=True))
    
    plan = WorkflowPlan(goal="Goal", steps=[
        WorkflowStep(step_id="1", name="A", agent_role="A", dependencies=[])
    ])
    
    res = ex.execute_plan(plan)
    assert res.status == "failed"
    assert "1" in res.failed_steps
