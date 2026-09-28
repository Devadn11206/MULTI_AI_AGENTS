"""
Phase 8/9: Model Factory
"""
from typing import Any, Dict, Optional
from core.model_engineering.registry import get_registry

class ModelFactory:
    """
    Creates un-fitted scikit-learn or XGBoost model instances.
    """
    
    @classmethod
    def create(cls, model_id: str, random_state: Optional[int] = None, **kwargs) -> Any:
        registry = get_registry()
        meta = registry.get_model(model_id)
        
        if not meta:
            raise ValueError(f"Unknown model_id: {model_id}")
            
        if not registry.is_available(model_id):
            raise ImportError(f"Model {model_id} dependencies are missing: {meta.dependencies}")
            
        params = kwargs.copy()
        if random_state is not None:
            # Some models don't support random_state, we only add it if supported
            if model_id not in ["linear_regression", "svm_classifier", "svm_regressor"]:
                params["random_state"] = random_state

        if model_id == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)
        elif model_id == "random_forest_classifier":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)
        elif model_id == "gradient_boosting_classifier":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**params)
        elif model_id == "xgboost_classifier":
            import xgboost as xgb
            return xgb.XGBClassifier(**params)
        elif model_id == "svm_classifier":
            from sklearn.svm import SVC
            if "probability" not in params:
                params["probability"] = True
            return SVC(**params)
        elif model_id == "linear_regression":
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**params)
        elif model_id == "random_forest_regressor":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**params)
        elif model_id == "gradient_boosting_regressor":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**params)
        elif model_id == "xgboost_regressor":
            import xgboost as xgb
            return xgb.XGBRegressor(**params)
        elif model_id == "svm_regressor":
            from sklearn.svm import SVR
            return SVR(**params)
        else:
            raise NotImplementedError(f"Model {model_id} initialization not implemented.")

