import joblib
import pandas as pd
import os
from core.ml.models import EvaluationResult, TrainingResult
from tools.ml.metric_calculator import MetricCalculator

class ModelEvaluator:
    """Handles generating predictions and evaluating model artifacts."""
    
    def __init__(self, output_dir: str = "artifacts/predictions"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def evaluate(
        self,
        training_result: TrainingResult,
        primary_metric: str,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> EvaluationResult:
        
        result = EvaluationResult(
            experiment_id=training_result.experiment_id,
            model_id=training_result.model_id,
            model_name=training_result.model_name,
            task_type=training_result.task_type,
            primary_metric=primary_metric,
            status="pending"
        )
        
        try:
            if training_result.status != "completed" or not training_result.artifact_path:
                raise ValueError("Cannot evaluate model that failed training or lacks artifact.")
                
            # Load model
            model = joblib.load(training_result.artifact_path)
            
            # Predict
            y_pred = model.predict(X_test)
            
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                except Exception:
                    pass
                
            # Calculate metrics
            if "classification" in training_result.task_type:
                is_multi = training_result.task_type == "multiclass_classification"
                metrics = MetricCalculator.calculate_classification_metrics(y_test, y_pred, y_prob, is_multi)
            else:
                metrics = MetricCalculator.calculate_regression_metrics(y_test, y_pred)
                
            # Store numeric metrics in dict
            result.metrics = {k: v.value for k, v in metrics.items() if v.status == "completed"}
            
            prim_lower = primary_metric.lower().replace("-", "_").replace(" ", "_")
            
            # Map common names
            if prim_lower in ["r2_score", "r_squared", "r_2"]:
                prim_lower = "r2"
            
            if prim_lower in result.metrics:
                result.primary_metric_value = result.metrics[prim_lower]
            else:
                result.warnings.append(f"Primary metric '{primary_metric}' not available.")
                
            # Save predictions
            pred_df = pd.DataFrame({"actual": y_test, "predicted": y_pred})
            pred_path = os.path.join(self.output_dir, f"{training_result.experiment_id}_{training_result.model_id}_preds.csv")
            pred_path = pred_path.replace("\\", "/")
            pred_df.to_csv(pred_path, index=False)
            
            result.prediction_artifact = pred_path
            result.status = "completed"
            
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            
        return result
