"""
Phase 7 unit tests — Feature Engineering Agent
All tests are deterministic. No real API keys required.
"""
import os
import uuid
import pytest
import pandas as pd

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIXTURE = "tests/fixtures/feature_engineering_dataset.csv"
FIXTURE_FULL = os.path.join(WORKSPACE, FIXTURE)
TARGET = "Churn"


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def load_fixture() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_FULL)


def make_simple_plan(target=TARGET, ops=None, removal=None, dup="keep"):
    from core.feature_engineering.models import FeatureEngineeringPlan
    return FeatureEngineeringPlan(
        target_column=target,
        operations=ops or [],
        feature_removal=removal or [],
        duplicate_handling=dup,
    )


# ────────────────────────────────────────────────────────────────────────────
# FeatureEngineeringPlan model tests
# ────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineeringPlan:
    def test_valid_plan(self):
        from core.feature_engineering.models import FeatureEngineeringPlan, TransformationOperation
        plan = FeatureEngineeringPlan(
            target_column=TARGET,
            operations=[
                TransformationOperation(operation="impute_numeric", columns=["TotalCharges"], strategy="median"),
            ],
        )
        assert plan.target_column == TARGET
        assert len(plan.operations) == 1

    def test_empty_plan(self):
        from core.feature_engineering.models import FeatureEngineeringPlan
        plan = FeatureEngineeringPlan(target_column=TARGET)
        assert plan.operations == []
        assert plan.feature_removal == []


# ────────────────────────────────────────────────────────────────────────────
# FeatureTransformer tests
# ────────────────────────────────────────────────────────────────────────────

class TestFeatureTransformerNumericImputation:
    def setup_method(self):
        from tools.feature_transformer import FeatureTransformer
        self.df = load_fixture()
        self.tr = FeatureTransformer(target_column=TARGET)

    def test_median_imputation(self):
        from core.feature_engineering.models import FeatureEngineeringPlan, TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(operation="impute_numeric", columns=["TotalCharges"], strategy="median")
        ])
        result_df, records = self.tr.fit_transform(self.df, plan)
        assert result_df["TotalCharges"].isna().sum() == 0
        assert any(r.operation == "impute_numeric" for r in records)

    def test_mean_imputation(self):
        from core.feature_engineering.models import FeatureEngineeringPlan, TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(operation="impute_numeric", columns=["TotalCharges"], strategy="mean")
        ])
        result_df, _ = self.tr.fit_transform(self.df, plan)
        assert result_df["TotalCharges"].isna().sum() == 0

    def test_target_not_imputed(self):
        from core.feature_engineering.models import TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(operation="impute_numeric", columns=[TARGET], strategy="median")
        ])
        # Should raise during transform (target guard)
        result_df, records = self.tr.fit_transform(self.df, plan)
        # Should have a failed record since target is protected
        failed = [r for r in records if r.status == "failed" and TARGET in (r.error or "")]
        assert len(failed) > 0 or result_df[TARGET].equals(self.df[TARGET])


class TestFeatureTransformerCategoricalImputation:
    def setup_method(self):
        from tools.feature_transformer import FeatureTransformer
        self.df = load_fixture()
        self.tr = FeatureTransformer(target_column=TARGET)

    def test_mode_imputation(self):
        from core.feature_engineering.models import TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(operation="impute_categorical", columns=["Gender"], strategy="mode")
        ])
        result_df, records = self.tr.fit_transform(self.df, plan)
        assert result_df["Gender"].isna().sum() == 0

    def test_constant_imputation(self):
        from core.feature_engineering.models import TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(operation="impute_categorical", columns=["Gender"],
                                    strategy="constant", constant_value="Unknown")
        ])
        result_df, _ = self.tr.fit_transform(self.df, plan)
        assert "Unknown" in result_df["Gender"].values
        assert result_df["Gender"].isna().sum() == 0


class TestFeatureTransformerEncoding:
    def setup_method(self):
        from tools.feature_transformer import FeatureTransformer
        self.df = load_fixture()
        self.tr = FeatureTransformer(target_column=TARGET)

    def test_one_hot_encode(self):
        from core.feature_engineering.models import TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(operation="one_hot_encode", columns=["City"])
        ])
        result_df, _ = self.tr.fit_transform(self.df, plan)
        assert "City" not in result_df.columns
        assert any(c.startswith("City_") for c in result_df.columns)

    def test_one_hot_does_not_encode_target(self):
        from core.feature_engineering.models import TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(operation="one_hot_encode", columns=[TARGET])
        ])
        result_df, records = self.tr.fit_transform(self.df, plan)
        # Target must survive unchanged
        assert TARGET in result_df.columns

    def test_binary_encode(self):
        from core.feature_engineering.models import TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(operation="binary_encode", columns=["Gender"])
        ])
        result_df, _ = self.tr.fit_transform(self.df, plan)
        # Gender has Male/Female — binary encode or one-hot fallback
        assert "Gender" in result_df.columns or any(c.startswith("Gender_") for c in result_df.columns)


class TestFeatureTransformerScaling:
    def setup_method(self):
        from tools.feature_transformer import FeatureTransformer
        self.df = load_fixture()
        self.tr = FeatureTransformer(target_column=TARGET)

    def test_standard_scaling(self):
        from core.feature_engineering.models import TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(operation="scale", columns=["Age", "MonthlyCharges"], strategy="standard")
        ])
        result_df, records = self.tr.fit_transform(self.df, plan)
        # After StandardScaler, mean should be ~ 0
        assert abs(result_df["Age"].mean()) < 1.0
        assert any(r.operation == "scale" for r in records)

    def test_minmax_scaling(self):
        from core.feature_engineering.models import TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(operation="scale", columns=["Age"], strategy="minmax")
        ])
        result_df, _ = self.tr.fit_transform(self.df, plan)
        # After MinMaxScaler, values should be in [0, 1]
        assert result_df["Age"].min() >= 0.0 - 1e-9
        assert result_df["Age"].max() <= 1.0 + 1e-9

    def test_target_not_scaled(self):
        from core.feature_engineering.models import TransformationOperation
        original_target = self.df[TARGET].copy()
        plan = make_simple_plan(ops=[
            TransformationOperation(operation="scale", columns=[TARGET], strategy="standard")
        ])
        result_df, records = self.tr.fit_transform(self.df, plan)
        failed = [r for r in records if r.status == "failed"]
        assert len(failed) > 0 or result_df[TARGET].equals(original_target)


class TestFeatureTransformerDuplicates:
    def setup_method(self):
        from tools.feature_transformer import FeatureTransformer
        self.df = load_fixture()
        self.tr = FeatureTransformer(target_column=TARGET)

    def test_remove_duplicates(self):
        plan = make_simple_plan(dup="remove")
        result_df, records = self.tr.fit_transform(self.df, plan)
        assert len(result_df) < len(self.df)
        assert result_df.duplicated().sum() == 0
        assert any(r.operation == "remove_duplicates" for r in records)

    def test_keep_duplicates(self):
        plan = make_simple_plan(dup="keep")
        result_df, records = self.tr.fit_transform(self.df, plan)
        # Row count must be unchanged
        assert len(result_df) == len(self.df)


class TestFeatureTransformerFeatureCreation:
    def setup_method(self):
        from tools.feature_transformer import FeatureTransformer
        self.df = load_fixture()
        self.tr = FeatureTransformer(target_column=TARGET)

    def test_create_ratio(self):
        from core.feature_engineering.models import TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(
                operation="create_ratio",
                numerator="MonthlyCharges",
                denominator="Tenure",
                new_column="charge_per_month"
            )
        ])
        result_df, _ = self.tr.fit_transform(self.df, plan)
        assert "charge_per_month" in result_df.columns

    def test_create_difference(self):
        from core.feature_engineering.models import TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(
                operation="create_difference",
                columns=["TotalCharges", "MonthlyCharges"],
                new_column="charge_diff"
            )
        ])
        result_df, _ = self.tr.fit_transform(self.df, plan)
        assert "charge_diff" in result_df.columns

    def test_create_sum(self):
        from core.feature_engineering.models import TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(
                operation="create_sum",
                columns=["MonthlyCharges", "Tenure"],
                new_column="charges_plus_tenure"
            )
        ])
        result_df, _ = self.tr.fit_transform(self.df, plan)
        assert "charges_plus_tenure" in result_df.columns

    def test_create_log(self):
        from core.feature_engineering.models import TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(operation="create_log", columns=["MonthlyCharges"], new_column="log_charges")
        ])
        result_df, _ = self.tr.fit_transform(self.df, plan)
        assert "log_charges" in result_df.columns
        assert result_df["log_charges"].isna().sum() == 0

    def test_create_ratio_zero_denominator(self):
        """Should produce NaN when denominator is 0, not crash."""
        from core.feature_engineering.models import TransformationOperation
        df = pd.DataFrame({"A": [1.0, 2.0], "B": [0.0, 1.0], "target": ["Y", "N"]})
        from tools.feature_transformer import FeatureTransformer
        tr = FeatureTransformer(target_column="target")
        plan = make_simple_plan(target="target", ops=[
            TransformationOperation(operation="create_ratio", numerator="A", denominator="B",
                                    new_column="ratio")
        ])
        result_df, records = tr.fit_transform(df, plan)
        assert "ratio" in result_df.columns
        assert result_df["ratio"].iloc[0] != result_df["ratio"].iloc[0]  # NaN check


class TestFeatureTransformerFeatureRemoval:
    def setup_method(self):
        from tools.feature_transformer import FeatureTransformer
        self.df = load_fixture()
        self.tr = FeatureTransformer(target_column=TARGET)

    def test_remove_id_column(self):
        plan = make_simple_plan(removal=["CustomerID"])
        result_df, _ = self.tr.fit_transform(self.df, plan)
        assert "CustomerID" not in result_df.columns

    def test_target_not_removed(self):
        """Target in feature_removal must be silently skipped."""
        plan = make_simple_plan(removal=[TARGET, "CustomerID"])
        result_df, _ = self.tr.fit_transform(self.df, plan)
        assert TARGET in result_df.columns

    def test_remove_column_op(self):
        from core.feature_engineering.models import TransformationOperation
        plan = make_simple_plan(ops=[
            TransformationOperation(operation="remove_column", columns=["CustomerID"])
        ])
        result_df, _ = self.tr.fit_transform(self.df, plan)
        assert "CustomerID" not in result_df.columns


class TestFeatureTransformerDatetime:
    def test_extract_datetime_features(self):
        from core.feature_engineering.models import TransformationOperation
        from tools.feature_transformer import FeatureTransformer
        df = pd.DataFrame({
            "signup_date": ["2023-01-15", "2023-06-20", "2022-12-01"],
            "target": ["Y", "N", "Y"]
        })
        tr = FeatureTransformer(target_column="target")
        plan = make_simple_plan(target="target", ops=[
            TransformationOperation(operation="extract_datetime_features", columns=["signup_date"])
        ])
        result_df, _ = tr.fit_transform(df, plan)
        assert "signup_date_year" in result_df.columns
        assert "signup_date_month" in result_df.columns
        assert "signup_date_day" in result_df.columns


# ────────────────────────────────────────────────────────────────────────────
# FeatureEngineeringAgent tests (mocked LLM)
# ────────────────────────────────────────────────────────────────────────────

def _make_mock_plan_response(target=TARGET):
    """Return a valid FeatureEngineeringPlan JSON string."""
    import json
    plan = {
        "target_column": target,
        "operations": [
            {"operation": "impute_numeric", "columns": ["TotalCharges"], "strategy": "median",
             "reason": "TotalCharges has missing values"},
            {"operation": "impute_categorical", "columns": ["Gender"], "strategy": "mode",
             "reason": "Gender has missing values"},
            {"operation": "one_hot_encode", "columns": ["City"],
             "reason": "City is categorical"},
            {"operation": "scale", "columns": ["Age", "MonthlyCharges", "Tenure"],
             "strategy": "standard", "reason": "Normalise numeric features"},
        ],
        "feature_removal": ["CustomerID"],
        "duplicate_handling": "remove",
        "reasoning": "Standard preprocessing for binary classification",
        "warnings": [],
    }
    return json.dumps(plan)


class MockLLMManager:
    provider_name = "mock"

    def generate(self, prompt, system_prompt=None, response_schema=None):
        from core.llm.models import LLMResponse
        return LLMResponse(
            content=_make_mock_plan_response(),
            provider="mock", model="mock"
        )


class TestFeatureEngineeringAgent:
    def setup_method(self):
        import unittest.mock as mock
        from config.settings import Settings
        self.patcher = mock.patch(
            "tools.dataset_reader.get_settings",
            return_value=Settings(_env_file=None, WORKSPACE_DIR=WORKSPACE)
        )
        self.patcher.start()
        from core.state.manager import AgentStateManager
        from agents.feature_engineering_agent import FeatureEngineeringAgent
        self.state = AgentStateManager()
        self.agent = FeatureEngineeringAgent(
            llm_provider=MockLLMManager(),
            state_manager=self.state,
            workspace_dir=WORKSPACE,
        )

    def teardown_method(self):
        self.patcher.stop()
        # Clean up any processed output
        import shutil
        processed = os.path.join(WORKSPACE, "data", "processed")
        if os.path.exists(processed):
            shutil.rmtree(processed)

    def test_engineer_returns_result(self):
        result = self.agent.engineer(FIXTURE, target_column=TARGET)
        assert result.status == "completed"

    def test_target_preserved(self):
        result = self.agent.engineer(FIXTURE, target_column=TARGET)
        assert result.target_column == TARGET

    def test_missing_values_reduced(self):
        result = self.agent.engineer(FIXTURE, target_column=TARGET)
        assert result.missing_after <= result.missing_before

    def test_duplicates_removed(self):
        result = self.agent.engineer(FIXTURE, target_column=TARGET)
        # Plan requests remove duplicates
        assert result.duplicates_after < result.duplicates_before

    def test_id_column_removed(self):
        result = self.agent.engineer(FIXTURE, target_column=TARGET)
        assert "CustomerID" in result.removed_features

    def test_one_hot_columns_added(self):
        result = self.agent.engineer(FIXTURE, target_column=TARGET)
        # City should have been one-hot encoded => City_* added
        city_cols = [c for c in result.transformed_columns if c.startswith("City_")]
        assert len(city_cols) > 0

    def test_processed_dataset_saved(self):
        result = self.agent.engineer(FIXTURE, target_column=TARGET)
        assert result.processed_dataset_path is not None
        full_path = os.path.join(WORKSPACE, result.processed_dataset_path)
        assert os.path.exists(full_path)

    def test_processed_dataset_loadable(self):
        result = self.agent.engineer(FIXTURE, target_column=TARGET)
        full_path = os.path.join(WORKSPACE, result.processed_dataset_path)
        df = pd.read_csv(full_path)
        assert len(df) > 0
        assert TARGET in df.columns

    def test_target_values_unchanged(self):
        df_original = load_fixture()
        result = self.agent.engineer(FIXTURE, target_column=TARGET)
        full_path = os.path.join(WORKSPACE, result.processed_dataset_path)
        df_transformed = pd.read_csv(full_path)
        # After duplicate removal, target distribution should be subset
        assert set(df_transformed[TARGET].unique()) == set(df_original[TARGET].unique())

    def test_original_not_overwritten(self):
        result = self.agent.engineer(FIXTURE, target_column=TARGET)
        # Original file must still exist and be unchanged
        assert os.path.exists(FIXTURE_FULL)
        df = pd.read_csv(FIXTURE_FULL)
        assert len(df) == 12  # Original row count

    def test_state_updated(self):
        self.agent.engineer(FIXTURE, target_column=TARGET)
        st = self.state.get_state()
        assert st.feature_engineering_result is not None
        assert st.processed_dataset_reference is not None

    def test_invalid_target_raises_error(self):
        result = self.agent.engineer(FIXTURE, target_column="NonExistentColumn")
        assert result.status == "failed"
        assert len(result.errors) > 0

    def test_transformations_recorded(self):
        result = self.agent.engineer(FIXTURE, target_column=TARGET)
        assert len(result.transformations) > 0

    def test_row_count_preserved_with_dedup(self):
        """Row count may decrease only when duplicates are removed."""
        df_original = load_fixture()
        result = self.agent.engineer(FIXTURE, target_column=TARGET)
        # We had 1 duplicate; after removal, count should be original-1
        assert result.transformed_shape[0] == df_original.shape[0] - 1


# ────────────────────────────────────────────────────────────────────────────
# Plan validation tests
# ────────────────────────────────────────────────────────────────────────────

class TestPlanValidation:
    def setup_method(self):
        import unittest.mock as mock
        from config.settings import Settings
        self.patcher = mock.patch(
            "tools.dataset_reader.get_settings",
            return_value=Settings(_env_file=None, WORKSPACE_DIR=WORKSPACE)
        )
        self.patcher.start()
        from core.state.manager import AgentStateManager
        from agents.feature_engineering_agent import FeatureEngineeringAgent
        self.agent = FeatureEngineeringAgent(
            llm_provider=MockLLMManager(),
            state_manager=AgentStateManager(),
            workspace_dir=WORKSPACE,
        )

    def teardown_method(self):
        self.patcher.stop()

    def test_unsupported_operation_rejected(self):
        from core.feature_engineering.models import FeatureEngineeringPlan, TransformationOperation
        df = load_fixture()
        plan = FeatureEngineeringPlan(
            target_column=TARGET,
            operations=[
                TransformationOperation(operation="train_model", columns=["Age"])
            ]
        )
        sanitized, warnings = self.agent._validate_plan(plan, df)
        assert len(sanitized.operations) == 0
        assert any("Unsupported" in w for w in warnings)

    def test_target_operation_rejected(self):
        from core.feature_engineering.models import FeatureEngineeringPlan, TransformationOperation
        df = load_fixture()
        plan = FeatureEngineeringPlan(
            target_column=TARGET,
            operations=[
                TransformationOperation(operation="scale", columns=[TARGET], strategy="standard")
            ]
        )
        sanitized, warnings = self.agent._validate_plan(plan, df)
        assert not any(TARGET in str(op.columns) for op in sanitized.operations)

    def test_target_removal_blocked(self):
        from core.feature_engineering.models import FeatureEngineeringPlan
        df = load_fixture()
        plan = FeatureEngineeringPlan(
            target_column=TARGET,
            feature_removal=[TARGET, "CustomerID"]
        )
        sanitized, warnings = self.agent._validate_plan(plan, df)
        assert TARGET not in sanitized.feature_removal

    def test_missing_column_rejected(self):
        from core.feature_engineering.models import FeatureEngineeringPlan, TransformationOperation
        df = load_fixture()
        plan = FeatureEngineeringPlan(
            target_column=TARGET,
            operations=[
                TransformationOperation(operation="scale", columns=["NonExistentColumn"])
            ]
        )
        sanitized, warnings = self.agent._validate_plan(plan, df)
        assert len(sanitized.operations) == 0
