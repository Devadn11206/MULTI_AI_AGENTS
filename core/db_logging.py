from core.database import SessionLocal
from repositories.repository import Repository

def log_agent_run_to_db(workflow_id: str, agent_name: str, status: str, error: str = None):
    db = SessionLocal()
    try:
        repo = Repository(db)
        repo.log_agent_run(workflow_id=workflow_id, agent_name=agent_name, status=status, error=error)
    finally:
        db.close()

def log_experiment_to_db_and_mlflow(workflow_id: str, model_name: str, task_type: str, metrics: dict, parameters: dict):
    from core.mlflow_integration import MLflowManager
    safe_model_name = str(model_name).strip() if model_name else "best_model"
    if safe_model_name == "None" or not safe_model_name:
        safe_model_name = "best_model"
    mlm = MLflowManager()
    run_id = mlm.log_experiment(workflow_id, safe_model_name, metrics, parameters, task_type)
    
    db = SessionLocal()
    try:
        repo = Repository(db)
        repo.log_experiment(workflow_id=workflow_id, model_name=safe_model_name, task_type=task_type, metrics=metrics, parameters=parameters, mlflow_run_id=run_id)
    finally:
        db.close()

