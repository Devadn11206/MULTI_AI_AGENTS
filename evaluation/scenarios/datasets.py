import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

def generate_datasets(output_dir: str = "data/eval"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Simple Classification
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_redundant=0, 
                               random_state=42, n_classes=2)
    df_simple = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    df_simple["target"] = y
    df_simple.to_csv(os.path.join(output_dir, "simple_classification.csv"), index=False)
    
    # 2. Medium Regression
    res = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    X, y = res[0], res[1]
    df_medium = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df_medium["target"] = y
    df_medium.to_csv(os.path.join(output_dir, "medium_regression.csv"), index=False)
    
    # 3. Messy Classification
    X, y = make_classification(n_samples=500, n_features=8, n_informative=4, random_state=42)
    df_messy = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
    df_messy["target"] = y
    
    # Add missing values
    for col in df_messy.columns[:-1]:
        mask = np.random.rand(len(df_messy)) < 0.1
        df_messy.loc[mask, col] = np.nan
        
    # Add categorical column
    df_messy["cat_feature"] = np.random.choice(["A", "B", "C", None], size=len(df_messy))
    df_messy.to_csv(os.path.join(output_dir, "messy_classification.csv"), index=False)
    
    # 4. Imbalanced Classification
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                               weights=[0.95, 0.05], random_state=42)
    df_imb = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df_imb["target"] = y
    df_imb.to_csv(os.path.join(output_dir, "imbalanced_classification.csv"), index=False)
    
    print(f"Generated 4 evaluation datasets in {output_dir}")

if __name__ == "__main__":
    generate_datasets()
