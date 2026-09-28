from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    non_null_count: int
    missing_count: int
    missing_percentage: float
    unique_count: int
    is_numeric: bool = False
    is_categorical: bool = False
    is_datetime: bool = False
    is_boolean: bool = False
    sample_values: List[Any] = Field(default_factory=list)
    # Numeric extras
    mean: Optional[float] = None
    median: Optional[float] = None
    standard_deviation: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    quartiles: Optional[Dict[str, float]] = None   # q1, q3, iqr
    # Categorical extras
    top_values: Optional[List[str]] = None
    value_counts: Optional[Dict[str, int]] = None


class TargetDistribution(BaseModel):
    target_column: str
    class_counts: Dict[str, int]
    class_percentages: Dict[str, float]
    number_of_classes: int
    imbalance_detected: bool


class OutlierInfo(BaseModel):
    column: str
    outlier_count: int
    outlier_percentage: float
    lower_bound: float
    upper_bound: float
    q1: float
    q3: float
    iqr: float


class CorrelationPair(BaseModel):
    column_a: str
    column_b: str
    correlation: float
    direction: str  # "positive" | "negative"


class DataWarning(BaseModel):
    severity: str      # "info" | "warning" | "critical"
    column: Optional[str] = None
    message: str


class DatasetProfile(BaseModel):
    dataset_path: str
    file_type: str
    row_count: int
    column_count: int
    columns: List[str] = Field(default_factory=list)
    data_types: Dict[str, str] = Field(default_factory=dict)
    column_profiles: Dict[str, ColumnProfile] = Field(default_factory=dict)

    # Missing values
    total_missing_values: int = 0
    columns_with_missing: List[str] = Field(default_factory=list)

    # Duplicates
    duplicate_count: int = 0
    duplicate_percentage: float = 0.0

    # Column categories
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)
    boolean_columns: List[str] = Field(default_factory=list)

    # Target
    target_column: Optional[str] = None
    target_distribution: Optional[TargetDistribution] = None

    # Outliers & Correlations
    outliers: List[OutlierInfo] = Field(default_factory=list)
    strong_correlations: List[CorrelationPair] = Field(default_factory=list)
    correlation_threshold: float = 0.70

    # Quality
    potential_id_columns: List[str] = Field(default_factory=list)
    constant_columns: List[str] = Field(default_factory=list)
    warnings: List[DataWarning] = Field(default_factory=list)

    # LLM-generated
    observations: Optional[str] = None
    recommendations: Optional[str] = None
