import os
import pytest
import pandas as pd
import numpy as np

from tools.ml.dataset_splitter import DatasetSplitter
from tools.ml.metric_calculator import MetricCalculator
from tools.ml.model_comparator import ModelComparator
from core.ml.models import EvaluationResult

def test_dataset_splitter():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    X_train, X_test, y_train, y_test = DatasetSplitter.split(
        df, "target", "binary_classification", test_size=0.2, random_state=42
    )
    
    assert len(X_train) == 8
    assert len(X_test) == 2

def test_metric_calculator_classification():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    
    metrics = MetricCalculator.calculate_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"].value == 0.75

def test_metric_calculator_regression():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]
    
    metrics = MetricCalculator.calculate_regression_metrics(y_true, y_pred)
    assert metrics["mse"].value == 0.0

def test_model_comparator():
    res1 = EvaluationResult(
        experiment_id="exp1", model_id="m1", model_name="M1", task_type="regression",
        primary_metric="rmse", primary_metric_value=0.5, status="completed"
    )
    res2 = EvaluationResult(
        experiment_id="exp2", model_id="m2", model_name="M2", task_type="regression",
        primary_metric="rmse", primary_metric_value=0.2, status="completed"
    )
    
    comp = ModelComparator.compare([res1, res2], "regression", "rmse")
    # For rmse, lower is better. So exp2 (0.2) should be best.
    assert comp.best_experiment_id == "exp2"

def test_model_comparator_higher_better():
    res1 = EvaluationResult(
        experiment_id="exp1", model_id="m1", model_name="M1", task_type="binary",
        primary_metric="recall", primary_metric_value=0.5, status="completed"
    )
    res2 = EvaluationResult(
        experiment_id="exp2", model_id="m2", model_name="M2", task_type="binary",
        primary_metric="recall", primary_metric_value=0.8, status="completed"
    )
    
    comp = ModelComparator.compare([res1, res2], "binary_classification", "recall")
    # For recall, higher is better. So exp2 (0.8) should be best.
    assert comp.best_experiment_id == "exp2"
