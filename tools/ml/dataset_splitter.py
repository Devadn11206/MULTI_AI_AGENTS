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
        
        actual_target = target_column
        if actual_target not in df.columns:
            # Try stripped / lowercase matching
            col_map = {c.strip().lower(): c for c in df.columns}
            target_key = target_column.strip().lower()
            if target_key in col_map:
                actual_target = col_map[target_key]
            else:
                raise ValueError(f"Target column '{target_column}' not found in dataset. Available: {list(df.columns)}")
            
        X = df.drop(columns=[actual_target])
        y = df[actual_target]
        
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
