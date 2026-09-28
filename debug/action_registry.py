from typing import Dict, Any, List

class ActionRegistry:
    def __init__(self):
        self._actions = {
            "class_weight": {
                "action_id": "class_weight",
                "description": "Applies class weighting to handle class imbalance.",
                "supported_tasks": ["classification"],
                "supported_models": ["logistic_regression", "random_forest_classifier", "svm_classifier"],
                "safety_level": "safe"
            },
            "threshold_optimization": {
                "action_id": "threshold_optimization",
                "description": "Optimizes decision threshold for binary classification.",
                "supported_tasks": ["classification"],
                "supported_models": ["logistic_regression", "random_forest_classifier", "gradient_boosting_classifier", "xgboost_classifier"],
                "safety_level": "safe"
            },
            "model_change": {
                "action_id": "model_change",
                "description": "Switches to another registered model.",
                "supported_tasks": ["classification", "regression"],
                "supported_models": ["*"],
                "safety_level": "safe"
            },
            "preprocessing_adjustment": {
                "action_id": "preprocessing_adjustment",
                "description": "Adjusts preprocessing steps such as scaling.",
                "supported_tasks": ["classification", "regression"],
                "supported_models": ["*"],
                "safety_level": "safe"
            },
            "metric_strategy_change": {
                "action_id": "metric_strategy_change",
                "description": "Changes the metric used for evaluation/optimization.",
                "supported_tasks": ["classification", "regression"],
                "supported_models": ["*"],
                "safety_level": "safe"
            },
            "cross_validation_adjustment": {
                "action_id": "cross_validation_adjustment",
                "description": "Adjusts cross validation folds.",
                "supported_tasks": ["classification", "regression"],
                "supported_models": ["*"],
                "safety_level": "safe"
            }
        }
        
    def get_action(self, action_id: str) -> Dict[str, Any]:
        return self._actions.get(action_id)

    def is_valid_action(self, action_id: str, task_type: str, model_id: str) -> bool:
        action = self.get_action(action_id)
        if not action:
            return False
            
        if task_type not in action["supported_tasks"]:
            return False
            
        if "*" not in action["supported_models"] and model_id not in action["supported_models"]:
            return False
            
        return True
