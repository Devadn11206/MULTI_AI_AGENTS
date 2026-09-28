"""
Pure Python/Pandas analysis functions.
The LLM is NEVER called here. All numbers come from Pandas.
"""
import math
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from core.analysis.models import (
    ColumnProfile, TargetDistribution, OutlierInfo,
    CorrelationPair, DataWarning, DatasetProfile
)

# Configuration constants
TOP_CATEGORY_VALUES = 10
MAX_SAMPLE_ROWS = 10
CORRELATION_THRESHOLD = 0.70
HIGH_MISSING_THRESHOLD = 0.20       # 20%
HIGH_OUTLIER_THRESHOLD = 0.10       # 10%
SMALL_DATASET_THRESHOLD = 50
IMBALANCE_THRESHOLD = 0.20          # minority class < 20%


def _safe_float(val) -> Optional[float]:
    """Convert numpy scalars to Python float, return None if NaN."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def analyze_shape(df: pd.DataFrame) -> Tuple[int, int]:
    return len(df), len(df.columns)


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def analyze_duplicates(df: pd.DataFrame) -> Tuple[int, float]:
    dup_count = int(df.duplicated().sum())
    dup_pct = round(dup_count / len(df) * 100, 2) if len(df) > 0 else 0.0
    return dup_count, dup_pct


# ---------------------------------------------------------------------------
# Data types classification
# ---------------------------------------------------------------------------

def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric, categorical, datetime_cols, boolean_cols = [], [], [], []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            boolean_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_cols.append(col)
        else:
            # Try parsing as datetime
            if df[col].dtype == object:
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df[col])
                    datetime_cols.append(col)
                    continue
                except Exception:
                    pass
            categorical.append(col)
    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_cols,
        "boolean": boolean_cols,
    }


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def analyze_missing_values(df: pd.DataFrame) -> Tuple[int, List[str], Dict[str, int]]:
    missing_by_col: Dict[str, int] = {
        col: int(df[col].isna().sum()) for col in df.columns
    }
    columns_with_missing = [c for c, v in missing_by_col.items() if v > 0]
    total_missing = sum(missing_by_col.values())
    return total_missing, columns_with_missing, missing_by_col


# ---------------------------------------------------------------------------
# Column profiles
# ---------------------------------------------------------------------------

def build_column_profile(col: str, df: pd.DataFrame, col_types: Dict) -> ColumnProfile:
    series = df[col]
    n = len(series)
    missing_count = int(series.isna().sum())
    non_null = n - missing_count
    missing_pct = round(missing_count / n * 100, 2) if n > 0 else 0.0
    unique_count = int(series.nunique(dropna=True))

    # Safe sample values
    sample = series.dropna().head(MAX_SAMPLE_ROWS).tolist()
    sample_values = [str(v) for v in sample]

    is_numeric = col in col_types.get("numeric", [])
    is_categorical = col in col_types.get("categorical", [])
    is_datetime = col in col_types.get("datetime", [])
    is_boolean = col in col_types.get("boolean", [])

    profile = ColumnProfile(
        name=col,
        dtype=str(series.dtype),
        non_null_count=non_null,
        missing_count=missing_count,
        missing_percentage=missing_pct,
        unique_count=unique_count,
        is_numeric=is_numeric,
        is_categorical=is_categorical,
        is_datetime=is_datetime,
        is_boolean=is_boolean,
        sample_values=sample_values,
    )

    if is_numeric:
        nums = series.dropna()
        q1 = _safe_float(nums.quantile(0.25)) if len(nums) > 0 else None
        q3 = _safe_float(nums.quantile(0.75)) if len(nums) > 0 else None
        iqr = round(q3 - q1, 4) if q1 is not None and q3 is not None else None
        profile.mean = _safe_float(nums.mean())
        profile.median = _safe_float(nums.median())
        profile.standard_deviation = _safe_float(nums.std())
        profile.minimum = _safe_float(nums.min())
        profile.maximum = _safe_float(nums.max())
        profile.quartiles = {"q1": q1, "q3": q3, "iqr": iqr} if q1 is not None else None

    if is_categorical or is_boolean:
        vc = series.value_counts(dropna=True).head(TOP_CATEGORY_VALUES)
        profile.top_values = [str(k) for k in vc.index.tolist()]
        profile.value_counts = {str(k): int(v) for k, v in vc.items()}

    return profile


# ---------------------------------------------------------------------------
# Target distribution
# ---------------------------------------------------------------------------

def analyze_target_distribution(df: pd.DataFrame, target_col: str) -> TargetDistribution:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    vc = df[target_col].value_counts(dropna=True)
    total = vc.sum()
    class_counts = {str(k): int(v) for k, v in vc.items()}
    class_pcts = {k: round(v / total * 100, 2) for k, v in class_counts.items()}
    n_classes = len(class_counts)
    minority_pct = min(class_pcts.values()) / 100 if class_pcts else 1.0
    imbalance = minority_pct < IMBALANCE_THRESHOLD

    return TargetDistribution(
        target_column=target_col,
        class_counts=class_counts,
        class_percentages=class_pcts,
        number_of_classes=n_classes,
        imbalance_detected=imbalance,
    )


# ---------------------------------------------------------------------------
# Outlier detection (IQR method)
# ---------------------------------------------------------------------------

def analyze_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> List[OutlierInfo]:
    results = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (series < lower) | (series > upper)
        count = int(outlier_mask.sum())
        pct = round(count / len(series) * 100, 2)
        results.append(OutlierInfo(
            column=col,
            outlier_count=count,
            outlier_percentage=pct,
            lower_bound=round(lower, 4),
            upper_bound=round(upper, 4),
            q1=round(q1, 4),
            q3=round(q3, 4),
            iqr=round(iqr, 4),
        ))
    return results


# ---------------------------------------------------------------------------
# Correlations (Pearson)
# ---------------------------------------------------------------------------

def analyze_correlations(
    df: pd.DataFrame,
    numeric_cols: List[str],
    threshold: float = CORRELATION_THRESHOLD,
) -> List[CorrelationPair]:
    if len(numeric_cols) < 2:
        return []
    try:
        corr_matrix = df[numeric_cols].corr(method="pearson")
    except Exception:
        return []

    pairs = []
    seen = set()
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1 :]:
            key = (col_a, col_b)
            if key in seen:
                continue
            seen.add(key)
            try:
                val = corr_matrix.loc[col_a, col_b]
                if pd.isna(val):
                    continue
                val = float(val)
                if abs(val) >= threshold:
                    pairs.append(CorrelationPair(
                        column_a=col_a,
                        column_b=col_b,
                        correlation=round(val, 4),
                        direction="positive" if val > 0 else "negative",
                    ))
            except Exception:
                continue
    pairs.sort(key=lambda p: abs(p.correlation), reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# ID column detection
# ---------------------------------------------------------------------------

def detect_id_columns(df: pd.DataFrame) -> List[str]:
    id_keywords = {"id", "uuid", "key", "code", "ref", "num", "no"}
    potential = []
    for col in df.columns:
        col_lower = col.lower()
        # Keyword heuristic
        if any(kw in col_lower for kw in id_keywords):
            potential.append(col)
            continue
        # High-cardinality string column that looks like identifiers
        if df[col].dtype == object:
            unique_ratio = df[col].nunique() / max(len(df), 1)
            if unique_ratio > 0.95:
                potential.append(col)
    return potential


# ---------------------------------------------------------------------------
# Constant / low-variance columns
# ---------------------------------------------------------------------------

def detect_constant_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if df[col].nunique(dropna=False) <= 1]


# ---------------------------------------------------------------------------
# Warnings generator
# ---------------------------------------------------------------------------

def generate_warnings(
    df: pd.DataFrame,
    profile: "DatasetProfile",
    missing_by_col: Dict[str, int],
) -> List[DataWarning]:
    n = len(df)
    warnings: List[DataWarning] = []

    # Empty dataset
    if n == 0:
        warnings.append(DataWarning(severity="critical", message="Dataset is empty (0 rows)."))
        return warnings

    # Very small dataset
    if n < SMALL_DATASET_THRESHOLD:
        warnings.append(DataWarning(
            severity="warning",
            message=f"Dataset is very small ({n} rows). Statistical results may not be reliable.",
        ))

    # High missingness per column
    for col, cnt in missing_by_col.items():
        pct = cnt / n
        if pct >= HIGH_MISSING_THRESHOLD:
            level = "critical" if pct >= 0.5 else "warning"
            warnings.append(DataWarning(
                severity=level, column=col,
                message=f"{round(pct*100,1)}% of values are missing in column '{col}'.",
            ))

    # Duplicate rows
    if profile.duplicate_count > 0:
        warnings.append(DataWarning(
            severity="warning",
            message=f"{profile.duplicate_count} duplicate rows detected ({profile.duplicate_percentage}%).",
        ))

    # Constant columns
    for col in profile.constant_columns:
        warnings.append(DataWarning(
            severity="warning", column=col,
            message=f"Column '{col}' has only one unique value (constant). May not be useful.",
        ))

    # Potential ID columns
    for col in profile.potential_id_columns:
        warnings.append(DataWarning(
            severity="info", column=col,
            message=f"Column '{col}' looks like an identifier. Consider excluding from modelling.",
        ))

    # Target imbalance
    if profile.target_distribution and profile.target_distribution.imbalance_detected:
        warnings.append(DataWarning(
            severity="warning",
            message=f"Class imbalance detected in target column '{profile.target_column}'.",
        ))

    # High outlier columns
    for o in profile.outliers:
        if o.outlier_percentage >= HIGH_OUTLIER_THRESHOLD * 100:
            warnings.append(DataWarning(
                severity="warning", column=o.column,
                message=f"Column '{o.column}' has {o.outlier_percentage}% outliers (IQR method).",
            ))

    return warnings
