from typing import Optional
from core.orchestration.models import WorkflowPlan, WorkflowStep

class Router:
    def get_next_step(self, plan: WorkflowPlan) -> Optional[WorkflowStep]:
        if plan.status in ["completed", "failed"]:
            return None
            
        if plan.failed_steps:
            plan.status = "failed"
            return None
            
        pending_steps = [s for s in plan.steps if s.status == "pending"]
        if not pending_steps:
            running = [s for s in plan.steps if s.status == "running"]
            if not running:
                plan.status = "completed"
            return None
            
        completed_set = set(plan.completed_steps)
        for step in pending_steps:
            if all(dep in completed_set for dep in step.dependencies):
                step.status = "ready"
                return step
                
        running_steps = [s for s in plan.steps if s.status == "running"]
        if not running_steps:
            raise ValueError("Workflow dependency deadlock detected. No steps can be executed.")
            
        return None
