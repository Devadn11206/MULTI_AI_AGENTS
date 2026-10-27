from core.state.manager import AgentStateManager
from core.orchestration.models import WorkflowPlan, WorkflowResult
from core.orchestration.router import Router
from agents.worker_agent import WorkerAgent
from agents.handoff import WorkerTask
from config.settings import get_settings
from typing import Optional

class Executor:
    def __init__(self, state_manager: AgentStateManager, worker_agent: WorkerAgent,
                 data_analyst_agent=None, feature_engineering_agent=None,
                 model_engineering_agent=None, training_agent=None, evaluation_agent=None):
        self.state_manager = state_manager
        self.worker_agent = worker_agent
        self.data_analyst_agent = data_analyst_agent
        self.feature_engineering_agent = feature_engineering_agent
        self.model_engineering_agent = model_engineering_agent
        self.training_agent = training_agent
        self.evaluation_agent = evaluation_agent
        self.router = Router()
        self.settings = get_settings()

    def execute_plan(self, plan: WorkflowPlan) -> WorkflowResult:
        plan.status = "running"
        iterations = 0
        # Allow enough iterations to execute each step plus retries
        cfg_max = getattr(self.settings, "MAX_WORKFLOW_ITERATIONS", 3)
        max_iter = max(cfg_max, len(plan.steps) * 3, 15)
        
        while plan.status == "running" and iterations < max_iter:
            iterations += 1
            try:
                step = self.router.get_next_step(plan)
            except Exception as e:
                plan.status = "failed"
                return WorkflowResult(
                    plan_id=plan.plan_id, status="failed",
                    completed_steps=plan.completed_steps, failed_steps=plan.failed_steps,
                    skipped_steps=[], final_message=str(e), errors=[str(e)]
                )
                
            if not step:
                break
                
            print(f"[Router] {step.step_id} ready")
            step.status = "running"
            plan.current_step = step.step_id
            
            agent_role = step.agent_role
            
            # Route data_analyst role to DataAnalystAgent (Phase 6)
            if agent_role == "data_analyst" and self.data_analyst_agent is not None:
                print(f"[Orchestrator] Assigning to: DataAnalystAgent (Role: {agent_role})")
                print(f"[DataAgent] Executing: {step.name}")
                step_info = {
                    "step_id": step.step_id,
                    "dataset_path": (self.state_manager.state.ml_problem or {}).get("dataset_path", ""),
                    "target_column": (self.state_manager.state.ml_problem or {}).get("target_column"),
                }
                self.state_manager.add_decision(action="step_started", reason=f"Started step {step.step_id}")
                try:
                    res = self.data_analyst_agent.execute_task_from_step(step_info)
                    if res.status == "completed":
                        step.status = "completed"
                        step.result = res.result
                        plan.completed_steps.append(step.step_id)
                        self.state_manager.add_decision(action="step_completed", reason=f"Step {step.step_id} completed")
                        if getattr(self.state_manager.state, "workflow_id", None):
                            from core.checkpoint_manager import CheckpointManager
                            CheckpointManager().save_checkpoint(self.state_manager.state.workflow_id, "data_analyst", "data_analyst", self.state_manager.state)
                        if getattr(self.state_manager.state, 'workflow_id', None):
                            from core.checkpoint_manager import CheckpointManager
                            CheckpointManager().save_checkpoint(self.state_manager.state.workflow_id, "data_analyst", "data_analyst", self.state_manager.state)
                    else:
                        step.status = "failed"
                        step.error = str(res.errors)
                        plan.failed_steps.append(step.step_id)
                        plan.status = "failed"
                        self.state_manager.add_decision(action="step_failed", reason=f"Step {step.step_id} failed: {res.errors}")
                except Exception as e:
                    step.status = "failed"
                    step.error = str(e)
                    plan.failed_steps.append(step.step_id)
                    plan.status = "failed"
                    self.state_manager.add_decision(action="step_failed", reason=f"Step {step.step_id} exception: {e}")
                continue

            # Route feature_engineer role to FeatureEngineeringAgent (Phase 7)
            if agent_role == "feature_engineer" and self.feature_engineering_agent is not None:
                print(f"[Orchestrator] Assigning to: FeatureEngineeringAgent (Role: {agent_role})")
                print(f"[FEAgent] Executing: {step.name}")
                ml_problem = self.state_manager.state.ml_problem or {}
                step_info = {
                    "step_id": step.step_id,
                    "dataset_path": ml_problem.get("dataset_path", ""),
                    "target_column": ml_problem.get("target_column", ""),
                }
                self.state_manager.add_decision(action="step_started", reason=f"Started step {step.step_id}")
                try:
                    res = self.feature_engineering_agent.execute_task_from_step(step_info)
                    if res.status == "completed":
                        step.status = "completed"
                        step.result = res.result
                        plan.completed_steps.append(step.step_id)
                        self.state_manager.add_decision(action="step_completed", reason=f"Step {step.step_id} completed")
                        if getattr(self.state_manager.state, "workflow_id", None):
                            from core.checkpoint_manager import CheckpointManager
                            CheckpointManager().save_checkpoint(self.state_manager.state.workflow_id, "feature_engineering", "feature_engineering", self.state_manager.state)
                    else:
                        step.status = "failed"
                        step.error = str(res.errors)
                        plan.failed_steps.append(step.step_id)
                        plan.status = "failed"
                        self.state_manager.add_decision(action="step_failed", reason=f"Step {step.step_id} failed: {res.errors}")
                except Exception as e:
                    step.status = "failed"
                    step.error = str(e)
                    plan.failed_steps.append(step.step_id)
                    plan.status = "failed"
                    self.state_manager.add_decision(action="step_failed", reason=f"Step {step.step_id} exception: {e}")
                continue

            # Route model_engineer role to ModelEngineeringAgent (Phase 8)
            if agent_role == "model_engineer" and self.model_engineering_agent is not None:
                print(f"[Orchestrator] Assigning to: ModelEngineeringAgent (Role: {agent_role})")
                print(f"[MEAgent] Executing: {step.name}")
                ml_problem = self.state_manager.state.ml_problem or {}
                step_info = {
                    "step_id": step.step_id,
                    "ml_problem": ml_problem.get("problem_description", ""),
                    "target_column": ml_problem.get("target_column", ""),
                    "task_type": ml_problem.get("task_type", ""),
                    "constraints": ml_problem.get("constraints", {})
                }
                self.state_manager.add_decision(action="step_started", reason=f"Started step {step.step_id}")
                try:
                    res = self.model_engineering_agent.execute_task_from_step(step_info)
                    if res.status == "completed":
                        step.status = "completed"
                        step.result = res.result
                        plan.completed_steps.append(step.step_id)
                        self.state_manager.add_decision(action="step_completed", reason=f"Step {step.step_id} completed")
                        if getattr(self.state_manager.state, "workflow_id", None):
                            from core.checkpoint_manager import CheckpointManager
                            CheckpointManager().save_checkpoint(self.state_manager.state.workflow_id, "model_engineering", "model_engineering", self.state_manager.state)
                    else:
                        step.status = "failed"
                        step.error = str(res.errors)
                        plan.failed_steps.append(step.step_id)
                        plan.status = "failed"
                        self.state_manager.add_decision(action="step_failed", reason=f"Step {step.step_id} failed: {res.errors}")
                except Exception as e:
                    step.status = "failed"
                    step.error = str(e)
                    plan.failed_steps.append(step.step_id)
                    plan.status = "failed"
                    self.state_manager.add_decision(action="step_failed", reason=f"Step {step.step_id} exception: {e}")
                continue

            # Route training role to TrainingAgent (Phase 9)
            if agent_role == "training" and self.training_agent is not None:
                print(f"[Orchestrator] Assigning to: TrainingAgent (Role: {agent_role})")
                print(f"[TrainingAgent] Executing: {step.name}")
                step_info = {"step_id": step.step_id}
                self.state_manager.add_decision(action="step_started", reason=f"Started step {step.step_id}")
                try:
                    res = self.training_agent.execute_task_from_step(step_info)
                    if res.status == "completed":
                        step.status = "completed"
                        step.result = res.result
                        plan.completed_steps.append(step.step_id)
                        self.state_manager.add_decision(action="step_completed", reason=f"Step {step.step_id} completed")
                        if getattr(self.state_manager.state, "workflow_id", None):
                            from core.checkpoint_manager import CheckpointManager
                            CheckpointManager().save_checkpoint(self.state_manager.state.workflow_id, "training", "training", self.state_manager.state)
                        if getattr(self.state_manager.state, 'workflow_id', None):
                            from core.checkpoint_manager import CheckpointManager
                            CheckpointManager().save_checkpoint(self.state_manager.state.workflow_id, "training", "training", self.state_manager.state)
                    else:
                        step.status = "failed"
                        step.error = str(res.errors)
                        plan.failed_steps.append(step.step_id)
                        plan.status = "failed"
                        self.state_manager.add_decision(action="step_failed", reason=f"Step {step.step_id} failed: {res.errors}")
                except Exception as e:
                    step.status = "failed"
                    step.error = str(e)
                    plan.failed_steps.append(step.step_id)
                    plan.status = "failed"
                    self.state_manager.add_decision(action="step_failed", reason=f"Step {step.step_id} exception: {e}")
                continue

            # Route evaluation role to EvaluationAgent (Phase 9)
            if agent_role == "evaluation" and self.evaluation_agent is not None:
                print(f"[Orchestrator] Assigning to: EvaluationAgent (Role: {agent_role})")
                print(f"[EvaluationAgent] Executing: {step.name}")
                step_info = {"step_id": step.step_id}
                self.state_manager.add_decision(action="step_started", reason=f"Started step {step.step_id}")
                try:
                    res = self.evaluation_agent.execute_task_from_step(step_info)
                    if res.status == "completed":
                        step.status = "completed"
                        step.result = res.result
                        plan.completed_steps.append(step.step_id)
                        self.state_manager.add_decision(action="step_completed", reason=f"Step {step.step_id} completed")
                        if getattr(self.state_manager.state, "workflow_id", None):
                            from core.checkpoint_manager import CheckpointManager
                            CheckpointManager().save_checkpoint(self.state_manager.state.workflow_id, "evaluation", "evaluation", self.state_manager.state)
                        if getattr(self.state_manager.state, 'workflow_id', None):
                            from core.checkpoint_manager import CheckpointManager
                            CheckpointManager().save_checkpoint(self.state_manager.state.workflow_id, "evaluation", "evaluation", self.state_manager.state)
                    else:
                        step.status = "failed"
                        step.error = str(res.errors)
                        plan.failed_steps.append(step.step_id)
                        plan.status = "failed"
                        self.state_manager.add_decision(action="step_failed", reason=f"Step {step.step_id} failed: {res.errors}")
                except Exception as e:
                    step.status = "failed"
                    step.error = str(e)
                    plan.failed_steps.append(step.step_id)
                    plan.status = "failed"
                    self.state_manager.add_decision(action="step_failed", reason=f"Step {step.step_id} exception: {e}")
                continue
            
            # Default: delegate to WorkerAgent
            print(f"[Orchestrator] Assigning to: WorkerAgent (Role: {agent_role})")
            print(f"[Worker] Executing: {step.name}")
            task = WorkerTask(
                task=f"Execute workflow step: {step.name}. Instructions: {step.description or 'None'}",
                context=self.state_manager.build_context()
            )
            
            self.state_manager.add_decision(action="step_started", reason=f"Started step {step.step_id}")
            
            try:
                res = self.worker_agent.execute_task(task)
                
                if res.status == "completed":
                    print(f"[Worker] Result: Simulation completed — specialized {agent_role} Agent will be connected in Phase 6.")
                    print(f"[Worker] Step completed")
                    step.status = "completed"
                    step.result = res.result
                    plan.completed_steps.append(step.step_id)
                    self.state_manager.add_decision(action="step_completed", reason=f"Step {step.step_id} completed")
                    if getattr(self.state_manager.state, "workflow_id", None):
                        from core.checkpoint_manager import CheckpointManager
                        CheckpointManager().save_checkpoint(self.state_manager.state.workflow_id, "worker", "worker", self.state_manager.state)
                else:
                    step.status = "failed"
                    step.error = str(res.errors)
                    plan.failed_steps.append(step.step_id)
                    plan.status = "failed"
                    self.state_manager.add_decision(action="step_failed", reason=f"Step {step.step_id} failed: {res.errors}")
            except Exception as e:
                step.status = "failed"
                step.error = str(e)
                plan.failed_steps.append(step.step_id)
                plan.status = "failed"
                self.state_manager.add_decision(action="step_failed", reason=f"Step {step.step_id} failed exception: {e}")
                
        if iterations >= max_iter and plan.status == "running":
            plan.status = "failed"
            return WorkflowResult(
                plan_id=plan.plan_id, status="failed",
                completed_steps=plan.completed_steps, failed_steps=plan.failed_steps,
                skipped_steps=[], final_message="MAX_WORKFLOW_ITERATIONS reached", errors=["Iteration limit reached"]
            )
            
        all_steps = {s.step_id for s in plan.steps}
        processed = set(plan.completed_steps + plan.failed_steps)
        skipped = list(all_steps - processed)
        for s in plan.steps:
            if s.step_id in skipped:
                s.status = "skipped"
                
        if not plan.failed_steps and plan.status == "running":
            plan.status = "completed"
                
        return WorkflowResult(
            plan_id=plan.plan_id,
            status=plan.status,
            completed_steps=plan.completed_steps,
            failed_steps=plan.failed_steps,
            skipped_steps=skipped,
            final_message="Workflow finished",
            errors=[]
        )
