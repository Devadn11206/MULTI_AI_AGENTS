import time
import os
import joblib
from core.ml.models import TrainingResult
from core.model_engineering.factory import ModelFactory

class ModelTrainer:
    """Handles model initialization, training, and artifact persistence."""
    
    def __init__(self, output_dir: str = "artifacts/models"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def train(
        self,
        experiment_id: str,
        model_id: str,
        model_name: str,
        task_type: str,
        X_train, 
        y_train, 
        X_test,
        random_state: int
    ) -> TrainingResult:
        
        start_time = time.time()
        result = TrainingResult(
            experiment_id=experiment_id,
            model_id=model_id,
            model_name=model_name,
            task_type=task_type,
            status="pending",
            train_rows=len(X_train),
            test_rows=len(X_test),
            feature_count=X_train.shape[1]
        )
        
        try:
            # Initialize model via Factory
            model = ModelFactory.create(model_id, random_state=random_state)
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Save artifact
            artifact_path = os.path.join(self.output_dir, f"{experiment_id}_{model_id}.joblib")
            # Replace backslashes just in case on windows
            artifact_path = artifact_path.replace("\\", "/")
            joblib.dump(model, artifact_path)
            
            result.artifact_path = artifact_path
            result.status = "completed"
            
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            
        result.training_time_seconds = round(time.time() - start_time, 2)
        return result
