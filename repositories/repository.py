from sqlalchemy.orm import Session
from models.domain import Workflow, AgentRun, Experiment, Dataset

class Repository:
    def __init__(self, db: Session):
        self.db = db

    def get_workflow(self, workflow_id: str, user_id: str) -> Workflow:
        return self.db.query(Workflow).filter(Workflow.id == workflow_id, Workflow.user_id == user_id).first()
        
    def create_workflow(self, user_id: str, dataset_id: str, objective: str, task_type: str, target_column: str, priority: str, workflow_id: str = None) -> Workflow:
        wf = Workflow(
            id=workflow_id,
            user_id=user_id,
            dataset_id=dataset_id,
            objective=objective,
            task_type=task_type,
            target_column=target_column,
            priority=priority,
            status="QUEUED"
        )
        self.db.add(wf)
        self.db.commit()
        self.db.refresh(wf)
        return wf

    def update_workflow_status(self, workflow_id: str, status: str, current_phase: str = None, error_message: str = None):
        wf = self.db.query(Workflow).filter(Workflow.id == workflow_id).first()
        if wf:
            wf.status = status
            if current_phase:
                wf.current_phase = current_phase
            if error_message:
                wf.error_message = error_message
            if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                from datetime import datetime
                wf.completed_at = datetime.utcnow()
            self.db.commit()
            
    def log_agent_run(self, workflow_id: str, agent_name: str, status: str, provider: str = None, model: str = None, error: str = None, latency_ms: int = None):
        run = AgentRun(
            workflow_id=workflow_id,
            agent_name=agent_name,
            status=status,
            provider=provider,
            model=model,
            error=error,
            latency_ms=latency_ms
        )
        self.db.add(run)
        self.db.commit()

    def log_experiment(self, workflow_id: str, model_name: str, task_type: str, metrics: dict, parameters: dict, best_model: bool = False, mlflow_run_id: str = None):
        exp = Experiment(
            workflow_id=workflow_id,
            model_name=model_name,
            task_type=task_type,
            metrics=metrics,
            parameters=parameters,
            best_model=best_model,
            mlflow_run_id=mlflow_run_id
        )
        self.db.add(exp)
        self.db.commit()
