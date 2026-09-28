from typing import List
from core.ml.models import EvaluationResult, ModelComparisonResult

class ModelComparator:
    """Compares model evaluation results and ranks them."""
    
    LOWER_IS_BETTER = {"mae", "mse", "rmse", "logloss", "log_loss"}
    
    @classmethod
    def compare(
        cls,
        eval_results: List[EvaluationResult],
        task_type: str,
        primary_metric: str
    ) -> ModelComparisonResult:
        
        comp = ModelComparisonResult(
            task_type=task_type,
            primary_metric=primary_metric
        )
        
        valid_results = [r for r in eval_results if r.status == "completed" and r.primary_metric_value is not None]
        
        if not valid_results:
            comp.comparison_warnings.append("No successful models to compare.")
            return comp
            
        prim_lower = primary_metric.lower().replace("-", "_").replace(" ", "_")
        if prim_lower in ["r2_score", "r_squared", "r_2"]:
            prim_lower = "r2"
            
        is_lower_better = prim_lower in cls.LOWER_IS_BETTER
        
        # Sort results
        valid_results.sort(
            key=lambda r: r.primary_metric_value, 
            reverse=not is_lower_better
        )
        
        # Generate ranking
        comp.experiments = [r.experiment_id for r in valid_results]
        
        for i, r in enumerate(valid_results):
            comp.ranking.append({
                "rank": i + 1,
                "experiment_id": r.experiment_id,
                "model_name": r.model_name,
                "primary_metric_value": r.primary_metric_value,
                "all_metrics": r.metrics
            })
            
        comp.best_model = valid_results[0].model_name
        comp.best_experiment_id = valid_results[0].experiment_id
        
        # Tie detection
        if len(valid_results) > 1:
            best_val = valid_results[0].primary_metric_value
            second_val = valid_results[1].primary_metric_value
            diff = abs(best_val - second_val)
            if diff <= 0.001:
                comp.comparison_warnings.append(
                    f"Near tie between {valid_results[0].model_name} and {valid_results[1].model_name} "
                    f"(diff={diff:.4f}). Consider secondary metrics."
                )
                
        return comp
