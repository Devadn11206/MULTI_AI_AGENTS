import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

class DatasetSplitter:
    """Handles data splitting with optional stratification."""
    
    @staticmethod
    def split(
        df: pd.DataFrame, 
        target_column: str, 
        task_type: str, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
            
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        stratify = None
        if task_type in ["binary_classification", "multiclass_classification"]:
            # Only stratify if we have enough samples of each class
            counts = y.value_counts()
            if len(counts) > 1 and counts.min() >= 2:
                stratify = y
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        return X_train, X_test, y_train, y_test
