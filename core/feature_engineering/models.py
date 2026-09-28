"""
Phase 7 — Feature Engineering Models

All planning and result structures.
The LLM produces a FeatureEngineeringPlan.
Python executes it.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Individual transformation operation
# ---------------------------------------------------------------------------

class TransformationOperation(BaseModel):
    """A single approved transformation step in the plan."""
    operation: str          # e.g. "impute_numeric", "one_hot_encode", "scale", "remove_column"
    columns: List[str] = Field(default_factory=list)
    strategy: Optional[str] = None          # "median", "mean", "mode", "standard", "minmax"
    constant_value: Optional[Any] = None    # for constant-fill imputation
    new_column: Optional[str] = None        # for feature-creation ops
    numerator: Optional[str] = None         # ratio numerator
    denominator: Optional[str] = None       # ratio denominator
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Full plan (LLM output — validated before execution)
# ---------------------------------------------------------------------------

class FeatureEngineeringPlan(BaseModel):
    """
    Structured plan produced by the LLM after inspecting the DatasetProfile.
    Python executes this; the LLM never touches the DataFrame.
    """
    target_column: str
    operations: List[TransformationOperation] = Field(default_factory=list)
    feature_removal: List[str] = Field(default_factory=list)   # columns to drop
    duplicate_handling: str = "keep"                            # "keep" | "remove"
    reasoning: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-transformation execution record
# ---------------------------------------------------------------------------

class TransformationRecord(BaseModel):
    """Records what actually happened during execution of one operation."""
    operation: str
    input_columns: List[str] = Field(default_factory=list)
    output_columns: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    rows_affected: int = 0
    status: str = "completed"   # "completed" | "failed" | "skipped"
    reason: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------

class FeatureEngineeringResult(BaseModel):
    """Returned by FeatureEngineeringAgent to the Orchestrator."""
    status: str                         # "completed" | "failed"
    original_shape: List[int]           # [rows, cols]
    transformed_shape: List[int]        # [rows, cols]
    original_columns: List[str] = Field(default_factory=list)
    transformed_columns: List[str] = Field(default_factory=list)
    added_features: List[str] = Field(default_factory=list)
    removed_features: List[str] = Field(default_factory=list)
    transformations: List[TransformationRecord] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    plan: Optional[FeatureEngineeringPlan] = None
    target_column: str = ""
    processed_dataset_path: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    # Missing-value summary
    missing_before: int = 0
    missing_after: int = 0
    # Duplicate summary
    duplicates_before: int = 0
    duplicates_after: int = 0
    rows_removed_duplicates: int = 0
