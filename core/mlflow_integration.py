import os
from typing import Dict, Any, Optional
from config.settings import get_settings
from core.logging import get_logger

logger = get_logger(__name__)

class MLflowManager:
    def __init__(self):
        self.settings = get_settings()
        self.enabled = self.settings.MLFLOW_ENABLED
        self.uri = self.settings.MLFLOW_TRACKING_URI
        self.mlflow = None
        
        if self.enabled:
            try:
                import mlflow
                self.mlflow = mlflow
                self.mlflow.set_tracking_uri(self.uri)
                logger.info(f"MLflow initialized with URI: {self.uri}")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow, continuing without it. Error: {e}")
                self.enabled = False

    def log_experiment(self, workflow_id: str, model_name: str, metrics: Dict[str, float], parameters: Optional[Dict[str, Any]] = None, task_type: str = "auto", dataset_hash: str = ""):
        if not self.enabled or not self.mlflow:
            logger.info("MLflow tracking unavailable. Skipping experiment logging.")
            return None
            
        try:
            experiment_name = f"workflow_{workflow_id}"
            self.mlflow.set_experiment(experiment_name)
            
            with self.mlflow.start_run(run_name=model_name) as run:
                if parameters:
                    self.mlflow.log_params(parameters)
                
                if metrics:
                    self.mlflow.log_metrics(metrics)
                    
                self.mlflow.set_tags({
                    "workflow_id": workflow_id,
                    "task_type": task_type,
                    "model_name": model_name,
                    "dataset_hash": dataset_hash
                })
                
                logger.info(f"Successfully logged experiment {model_name} to MLflow run {run.info.run_id}")
                return run.info.run_id
                
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            return None
