import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from core.ml.models import MetricResult

class MetricCalculator:
    """Calculates deterministic classification and regression metrics."""
    
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred, y_prob=None, is_multiclass=False) -> Dict[str, MetricResult]:
        metrics = {}
        
        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        metrics["accuracy"] = MetricResult(metric_name="accuracy", value=round(float(acc), 4), dataset_split="test")
        
        avg = "macro" if is_multiclass else "weighted"
        
        # Precision, Recall, F1
        prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
        
        metrics["precision"] = MetricResult(metric_name="precision", value=round(float(prec), 4), dataset_split="test", averaging=avg)
        metrics["recall"] = MetricResult(metric_name="recall", value=round(float(rec), 4), dataset_split="test", averaging=avg)
        metrics["f1"] = MetricResult(metric_name="f1", value=round(float(f1), 4), dataset_split="test", averaging=avg)
        
        # ROC AUC
        if y_prob is not None:
            try:
                if is_multiclass:
                    auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
                else:
                    # For binary, y_prob might be 2D or 1D
                    if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
                        prob = y_prob[:, 1]
                    else:
                        prob = y_prob
                    auc = roc_auc_score(y_true, prob)
                metrics["roc_auc"] = MetricResult(metric_name="roc_auc", value=round(float(auc), 4), dataset_split="test")
            except Exception as e:
                metrics["roc_auc"] = MetricResult(metric_name="roc_auc", value=0.0, dataset_split="test", status="failed", error=str(e))
                
        return metrics

    @staticmethod
    def calculate_regression_metrics(y_true, y_pred) -> Dict[str, MetricResult]:
        metrics = {}
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics["mae"] = MetricResult(metric_name="mae", value=round(float(mae), 4), dataset_split="test")
        metrics["mse"] = MetricResult(metric_name="mse", value=round(float(mse), 4), dataset_split="test")
        metrics["rmse"] = MetricResult(metric_name="rmse", value=round(float(rmse), 4), dataset_split="test")
        metrics["r2"] = MetricResult(metric_name="r2", value=round(float(r2), 4), dataset_split="test")
        
        return metrics
