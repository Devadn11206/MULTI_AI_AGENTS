from core.llm_manager import LLMManager
from core.orchestration.models import MLProblem, WorkflowPlan

class Planner:
    def __init__(self, llm_provider: LLMManager):
        self.llm_provider = llm_provider
        self.system_prompt = (
            "You are the MLForge Planner. Convert the user's ML problem into a structured WorkflowPlan. "
            "Return a JSON object containing 'goal' and a list of 'steps'. Each step must have 'step_id', 'name', 'agent_role', and 'dependencies'. "
            "Dependencies is a list of step_ids this step depends on. Ensure no circular dependencies and each step_id is unique. "
            "CRITICAL: 'agent_role' MUST be exactly one of: 'data_analyst', 'feature_engineer', 'model_engineer', 'training', 'evaluation'. "
            "You MUST include at least a 'data_analyst', 'feature_engineer', and 'model_engineer' step."
        )

    def generate_plan(self, problem: MLProblem) -> WorkflowPlan:
        prompt = f"Problem Statement: {problem.problem_statement}\nConstraints: {problem.constraints}\nOptimization Metric: {problem.optimization_metric}"
        
        try:
            resp = self.llm_provider.generate(prompt, self.system_prompt, response_schema=WorkflowPlan)
            plan = WorkflowPlan.model_validate_json(resp.content)
            plan.problem = problem
            return plan
        except Exception as e:
            raise ValueError(f"Failed to generate plan: {e}")

    def validate_plan(self, plan: WorkflowPlan):
        if not plan.steps:
            raise ValueError("Plan contains no steps")
            
        step_ids = {step.step_id for step in plan.steps}
        if len(step_ids) != len(plan.steps):
            raise ValueError("Duplicate step IDs found")
            
        for step in plan.steps:
            if not step.agent_role:
                raise ValueError(f"Step {step.step_id} is missing agent_role")
            if step.step_id in step.dependencies:
                raise ValueError(f"Step {step.step_id} depends on itself")
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise ValueError(f"Step {step.step_id} depends on non-existent step {dep}")
                    
        # check circular dependencies
        visited = set()
        path = set()
        
        def visit(node_id):
            if node_id in path:
                raise ValueError("Circular dependency detected")
            if node_id in visited:
                return
            path.add(node_id)
            step = next(s for s in plan.steps if s.step_id == node_id)
            for dep in step.dependencies:
                visit(dep)
            path.remove(node_id)
            visited.add(node_id)
            
        for step in plan.steps:
            visit(step.step_id)
