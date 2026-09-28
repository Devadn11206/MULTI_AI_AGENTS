from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager
from agents.worker_agent import WorkerAgent
from core.orchestration.models import MLProblem, WorkflowResult
from core.orchestration.planner import Planner
from core.orchestration.executor import Executor

class Orchestrator:
    def __init__(self, llm_provider: LLMManager, state_manager: AgentStateManager,
                 worker_agent: WorkerAgent, data_analyst_agent=None,
                 feature_engineering_agent=None,
                 model_engineering_agent=None,
                 training_agent=None,
                 evaluation_agent=None):
        self.state_manager = state_manager
        self.planner = Planner(llm_provider)
        self.executor = Executor(
            state_manager, worker_agent,
            data_analyst_agent=data_analyst_agent,
            feature_engineering_agent=feature_engineering_agent,
            model_engineering_agent=model_engineering_agent,
            training_agent=training_agent,
            evaluation_agent=evaluation_agent
        )

    def run_workflow(self, problem: MLProblem) -> WorkflowResult:
        print("[Orchestrator] Creating ML problem")
        if hasattr(self.state_manager.state, 'ml_problem'):
            self.state_manager.state.ml_problem = problem.model_dump()
        self.state_manager.add_decision(action="workflow_created", reason="Received ML Problem")
        
        print("[Orchestrator] Generating workflow plan")
        try:
            plan = self.planner.generate_plan(problem)
            self.state_manager.add_decision(action="plan_generated", reason="Planner generated workflow plan")
            
            print("[Orchestrator] Validating plan")
            self.planner.validate_plan(plan)
            self.state_manager.add_decision(action="plan_validated", reason="Workflow plan passed validation")
        except Exception as e:
            print(f"[Orchestrator] Failed during planning: {e}")
            self.state_manager.add_decision(action="workflow_failed", reason=f"Planning failed: {e}")
            return WorkflowResult(plan_id="error", status="failed", final_message=str(e))
        if hasattr(self.state_manager.state, 'workflow_plan'):
            self.state_manager.state.workflow_plan = plan.model_dump()
        
        print("\nWORKFLOW PLAN\n")
        for step in plan.steps:
            print(f"{step.step_id}. {step.name}")
        print("\n")
        
        result = self.executor.execute_plan(plan)
        
        if hasattr(self.state_manager.state, 'workflow_result'):
            self.state_manager.state.workflow_result = result.model_dump()
        
        if result.status == "completed":
            print("[Orchestrator] Workflow completed")
            self.state_manager.add_decision(action="workflow_completed", reason="All steps executed successfully")
        else:
            print("[Orchestrator] Workflow failed")
            self.state_manager.add_decision(action="workflow_failed", reason=result.final_message)
            
        return result
