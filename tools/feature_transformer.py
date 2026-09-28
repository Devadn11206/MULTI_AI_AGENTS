"""
FeatureTransformer — Phase 7

Executes a FeatureEngineeringPlan on a Pandas DataFrame.

Rules:
- Always works on a COPY of the original DataFrame.
- Never executes arbitrary LLM-generated code.
- Each operation is a named, deterministic function.
- Target column is protected at every step.
"""
import os
import math
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from core.feature_engineering.models import (
    FeatureEngineeringPlan, TransformationOperation, TransformationRecord,
)

warnings.filterwarnings("ignore", category=FutureWarning)


SUPPORTED_OPERATIONS = {
    "impute_numeric",
    "impute_categorical",
    "one_hot_encode",
    "binary_encode",
    "scale",
    "remove_column",
    "remove_duplicates",
    "create_ratio",
    "create_difference",
    "create_sum",
    "create_product",
    "create_log",
    "extract_datetime_features",
}


class FeatureTransformer:
    """
    Executes a validated FeatureEngineeringPlan on a DataFrame.
    Does NOT touch the original DataFrame — works on a copy.
    """

    def __init__(self, target_column: str):
        self.target_column = target_column
        self.records: List[TransformationRecord] = []
        # Fitted transformers (for future train/test compatibility)
        self._scalers: Dict[str, Any] = {}
        self._encoders: Dict[str, Any] = {}
        self._impute_values: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame, plan: FeatureEngineeringPlan
    ) -> Tuple[pd.DataFrame, List[TransformationRecord]]:
        """
        Execute the plan on a fresh copy of `df`.
        Returns (transformed_df, list_of_records).
        """
        result = df.copy()
        self.records = []

        # 1. Remove explicit feature columns FIRST (but never target)
        if plan.feature_removal:
            result = self._apply_feature_removal(result, plan.feature_removal)

        # 2. Duplicate handling
        if plan.duplicate_handling == "remove":
            result = self._apply_remove_duplicates(result)

        # 3. Ordered operations from plan
        for op in plan.operations:
            try:
                result = self._dispatch(result, op)
            except Exception as e:
                self.records.append(TransformationRecord(
                    operation=op.operation,
                    input_columns=op.columns,
                    status="failed",
                    error=str(e),
                    reason=op.reason,
                ))

        return result, self.records

    def transform(
        self, df: pd.DataFrame, plan: FeatureEngineeringPlan
    ) -> Tuple[pd.DataFrame, List[TransformationRecord]]:
        """
        Apply previously-fitted transformers to a new DataFrame (test/val).
        For now, delegates to fit_transform since we don't split yet.
        """
        return self.fit_transform(df, plan)

    # ------------------------------------------------------------------
    # Operation dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, df: pd.DataFrame, op: TransformationOperation) -> pd.DataFrame:
        o = op.operation
        if o == "impute_numeric":
            return self._impute_numeric(df, op)
        elif o == "impute_categorical":
            return self._impute_categorical(df, op)
        elif o == "one_hot_encode":
            return self._one_hot_encode(df, op)
        elif o == "binary_encode":
            return self._binary_encode(df, op)
        elif o == "scale":
            return self._scale(df, op)
        elif o == "remove_column":
            return self._remove_single_columns(df, op)
        elif o == "remove_duplicates":
            return self._apply_remove_duplicates(df)
        elif o == "create_ratio":
            return self._create_ratio(df, op)
        elif o == "create_difference":
            return self._create_difference(df, op)
        elif o == "create_sum":
            return self._create_sum(df, op)
        elif o == "create_product":
            return self._create_product(df, op)
        elif o == "create_log":
            return self._create_log(df, op)
        elif o == "extract_datetime_features":
            return self._extract_datetime_features(df, op)
        else:
            raise ValueError(f"Unsupported operation: '{o}'")

    # ------------------------------------------------------------------
    # Guard: never touch the target column
    # ------------------------------------------------------------------

    def _guard_target(self, columns: List[str], op_name: str):
        if self.target_column in columns:
            raise ValueError(
                f"Operation '{op_name}' attempted to transform the target column "
                f"'{self.target_column}'. Target is protected."
            )

    def _existing_cols(self, df: pd.DataFrame, columns: List[str], op_name: str) -> List[str]:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Operation '{op_name}': columns not found in dataset: {missing}")
        return columns

    # ------------------------------------------------------------------
    # Missing-value imputation
    # ------------------------------------------------------------------

    def _impute_numeric(self, df: pd.DataFrame, op: TransformationOperation) -> pd.DataFrame:
        cols = self._existing_cols(df, op.columns, "impute_numeric")
        self._guard_target(cols, "impute_numeric")
        strategy = (op.strategy or "median").lower()
        rows_affected = 0
        for col in cols:
            n_missing = df[col].isna().sum()
            rows_affected += int(n_missing)
            if n_missing == 0:
                continue
            if strategy == "median":
                fill_val = df[col].median()
            elif strategy == "mean":
                fill_val = df[col].mean()
            elif strategy == "constant":
                fill_val = op.constant_value if op.constant_value is not None else 0
            else:
                fill_val = df[col].median()
            self._impute_values[col] = fill_val
            df[col] = df[col].fillna(fill_val)

        self.records.append(TransformationRecord(
            operation="impute_numeric",
            input_columns=cols,
            output_columns=cols,
            parameters={"strategy": strategy},
            rows_affected=rows_affected,
            reason=op.reason,
        ))
        return df

    def _impute_categorical(self, df: pd.DataFrame, op: TransformationOperation) -> pd.DataFrame:
        cols = self._existing_cols(df, op.columns, "impute_categorical")
        self._guard_target(cols, "impute_categorical")
        strategy = (op.strategy or "mode").lower()
        rows_affected = 0
        for col in cols:
            n_missing = df[col].isna().sum()
            rows_affected += int(n_missing)
            if n_missing == 0:
                continue
            if strategy == "mode":
                mode_vals = df[col].mode(dropna=True)
                fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"
            elif strategy == "constant":
                fill_val = str(op.constant_value) if op.constant_value is not None else "Unknown"
            else:
                mode_vals = df[col].mode(dropna=True)
                fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"
            self._impute_values[col] = fill_val
            df[col] = df[col].fillna(fill_val)

        self.records.append(TransformationRecord(
            operation="impute_categorical",
            input_columns=cols,
            output_columns=cols,
            parameters={"strategy": strategy},
            rows_affected=rows_affected,
            reason=op.reason,
        ))
        return df

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _one_hot_encode(self, df: pd.DataFrame, op: TransformationOperation) -> pd.DataFrame:
        cols = self._existing_cols(df, op.columns, "one_hot_encode")
        self._guard_target(cols, "one_hot_encode")
        new_cols: List[str] = []
        for col in cols:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
            new_cols.extend(dummies.columns.tolist())
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        self.records.append(TransformationRecord(
            operation="one_hot_encode",
            input_columns=cols,
            output_columns=new_cols,
            parameters={},
            rows_affected=len(df),
            reason=op.reason,
        ))
        return df

    def _binary_encode(self, df: pd.DataFrame, op: TransformationOperation) -> pd.DataFrame:
        """Binary encode (0/1) for columns with exactly 2 unique values."""
        cols = self._existing_cols(df, op.columns, "binary_encode")
        self._guard_target(cols, "binary_encode")
        for col in cols:
            unique_vals = df[col].dropna().unique().tolist()
            if len(unique_vals) != 2:
                # Fall back to one-hot if not binary
                df = self._one_hot_encode(df, TransformationOperation(
                    operation="one_hot_encode", columns=[col], reason=op.reason))
                continue
            df[col] = (df[col] == unique_vals[0]).astype(int)

        self.records.append(TransformationRecord(
            operation="binary_encode",
            input_columns=cols,
            output_columns=cols,
            parameters={},
            rows_affected=len(df),
            reason=op.reason,
        ))
        return df

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def _scale(self, df: pd.DataFrame, op: TransformationOperation) -> pd.DataFrame:
        cols = self._existing_cols(df, op.columns, "scale")
        self._guard_target(cols, "scale")
        strategy = (op.strategy or "standard").lower()

        valid_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not valid_cols:
            self.records.append(TransformationRecord(
                operation="scale", input_columns=cols, output_columns=cols,
                status="skipped", reason="No numeric columns to scale"))
            return df

        if strategy in ("standard", "standardscaler", "z-score"):
            scaler = StandardScaler()
        elif strategy in ("minmax", "min_max", "minmaxscaler"):
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        df[valid_cols] = scaler.fit_transform(df[valid_cols])
        self._scalers[strategy] = scaler

        self.records.append(TransformationRecord(
            operation="scale",
            input_columns=valid_cols,
            output_columns=valid_cols,
            parameters={"strategy": strategy},
            rows_affected=len(df),
            reason=op.reason,
        ))
        return df

    # ------------------------------------------------------------------
    # Feature removal
    # ------------------------------------------------------------------

    def _apply_feature_removal(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        safe = [c for c in columns if c != self.target_column and c in df.columns]
        skipped = [c for c in columns if c not in safe]
        if skipped:
            self.records.append(TransformationRecord(
                operation="remove_column",
                input_columns=skipped,
                output_columns=[],
                status="skipped",
                reason="Column not found or is target column",
            ))
        if safe:
            df = df.drop(columns=safe)
            self.records.append(TransformationRecord(
                operation="remove_column",
                input_columns=safe,
                output_columns=[],
                rows_affected=len(df),
                reason="Explicit feature removal from plan",
            ))
        return df

    def _remove_single_columns(self, df: pd.DataFrame, op: TransformationOperation) -> pd.DataFrame:
        return self._apply_feature_removal(df, op.columns)

    # ------------------------------------------------------------------
    # Duplicate removal
    # ------------------------------------------------------------------

    def _apply_remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        removed = before - after
        self.records.append(TransformationRecord(
            operation="remove_duplicates",
            input_columns=[],
            output_columns=[],
            parameters={"rows_removed": removed},
            rows_affected=removed,
            reason="Duplicate removal requested in plan",
        ))
        return df

    # ------------------------------------------------------------------
    # Feature creation
    # ------------------------------------------------------------------

    def _create_ratio(self, df: pd.DataFrame, op: TransformationOperation) -> pd.DataFrame:
        num = op.numerator
        den = op.denominator
        new_col = op.new_column or f"{num}_div_{den}"
        if not num or not den:
            raise ValueError("create_ratio requires 'numerator' and 'denominator'.")
        self._existing_cols(df, [num, den], "create_ratio")
        self._guard_target([num, den], "create_ratio")
        if new_col in df.columns:
            raise ValueError(f"Column '{new_col}' already exists. Choose a different name.")
        # Safe division — avoid zero-division by replacing 0s with NaN
        denom = df[den].replace(0, float("nan"))
        df[new_col] = df[num] / denom
        self.records.append(TransformationRecord(
            operation="create_ratio", input_columns=[num, den],
            output_columns=[new_col], rows_affected=len(df), reason=op.reason,
        ))
        return df

    def _create_difference(self, df: pd.DataFrame, op: TransformationOperation) -> pd.DataFrame:
        if len(op.columns) < 2:
            raise ValueError("create_difference requires exactly 2 columns.")
        col_a, col_b = op.columns[0], op.columns[1]
        new_col = op.new_column or f"{col_a}_minus_{col_b}"
        self._existing_cols(df, [col_a, col_b], "create_difference")
        self._guard_target([col_a, col_b], "create_difference")
        if new_col in df.columns:
            raise ValueError(f"Column '{new_col}' already exists.")
        df[new_col] = df[col_a] - df[col_b]
        self.records.append(TransformationRecord(
            operation="create_difference", input_columns=[col_a, col_b],
            output_columns=[new_col], rows_affected=len(df), reason=op.reason,
        ))
        return df

    def _create_sum(self, df: pd.DataFrame, op: TransformationOperation) -> pd.DataFrame:
        self._existing_cols(df, op.columns, "create_sum")
        self._guard_target(op.columns, "create_sum")
        new_col = op.new_column or "_plus_".join(op.columns)
        if new_col in df.columns:
            raise ValueError(f"Column '{new_col}' already exists.")
        df[new_col] = df[op.columns].sum(axis=1)
        self.records.append(TransformationRecord(
            operation="create_sum", input_columns=op.columns,
            output_columns=[new_col], rows_affected=len(df), reason=op.reason,
        ))
        return df

    def _create_product(self, df: pd.DataFrame, op: TransformationOperation) -> pd.DataFrame:
        self._existing_cols(df, op.columns, "create_product")
        self._guard_target(op.columns, "create_product")
        new_col = op.new_column or "_times_".join(op.columns)
        if new_col in df.columns:
            raise ValueError(f"Column '{new_col}' already exists.")
        df[new_col] = df[op.columns].prod(axis=1)
        self.records.append(TransformationRecord(
            operation="create_product", input_columns=op.columns,
            output_columns=[new_col], rows_affected=len(df), reason=op.reason,
        ))
        return df

    def _create_log(self, df: pd.DataFrame, op: TransformationOperation) -> pd.DataFrame:
        cols = self._existing_cols(df, op.columns, "create_log")
        self._guard_target(cols, "create_log")
        new_cols = []
        for col in cols:
            new_col = op.new_column or f"log_{col}"
            if new_col in df.columns:
                new_col = f"log_{col}_new"
            # log1p is safe for zero values
            df[new_col] = np.log1p(df[col].clip(lower=0))
            new_cols.append(new_col)
        self.records.append(TransformationRecord(
            operation="create_log", input_columns=cols,
            output_columns=new_cols, rows_affected=len(df), reason=op.reason,
        ))
        return df

    # ------------------------------------------------------------------
    # Datetime features
    # ------------------------------------------------------------------

    def _extract_datetime_features(self, df: pd.DataFrame, op: TransformationOperation) -> pd.DataFrame:
        cols = self._existing_cols(df, op.columns, "extract_datetime_features")
        self._guard_target(cols, "extract_datetime_features")
        new_cols = []
        for col in cols:
            try:
                dt = pd.to_datetime(df[col], errors="coerce")
                for attr in ["year", "month", "day", "dayofweek"]:
                    new_col = f"{col}_{attr}"
                    df[new_col] = getattr(dt.dt, attr)
                    new_cols.append(new_col)
            except Exception as e:
                raise ValueError(f"Could not parse '{col}' as datetime: {e}")
        self.records.append(TransformationRecord(
            operation="extract_datetime_features", input_columns=cols,
            output_columns=new_cols, rows_affected=len(df), reason=op.reason,
        ))
        return df
