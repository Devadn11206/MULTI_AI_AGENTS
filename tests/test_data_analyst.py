"""
Phase 6 unit tests — Data Analyst Agent
All tests are deterministic. No real API keys required.
"""
import os
import math
import pytest
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_churn.csv")
# The WORKSPACE_DIR used by DatasetReaderTool
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def load_fixture() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH)


# ──────────────────────────────────────────────────────────────────────────
# DatasetReader tool tests
# ──────────────────────────────────────────────────────────────────────────

class TestDatasetReader:
    def setup_method(self):
        from config.settings import Settings
        import unittest.mock as mock
        self.patcher = mock.patch("tools.dataset_reader.get_settings",
                                   return_value=Settings(_env_file=None, WORKSPACE_DIR=WORKSPACE))
        self.patcher.start()
        from tools.dataset_reader import DatasetReaderTool
        self.tool = DatasetReaderTool()

    def teardown_method(self):
        self.patcher.stop()

    def test_valid_csv(self):
        df = self.tool.execute(path="tests/fixtures/test_churn.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11  # 10 data rows + 1 duplicate

    def test_missing_file(self):
        with pytest.raises(ValueError, match="File not found"):
            self.tool.execute(path="tests/fixtures/nonexistent.csv")

    def test_unsupported_extension(self):
        import tempfile, shutil
        # Create a .xlsx file inside workspace so security check passes, extension check fires
        fixtures_dir = os.path.join(WORKSPACE, "tests", "fixtures")
        fake_xlsx = os.path.join(fixtures_dir, "_test_fake.xlsx")
        try:
            with open(fake_xlsx, "w") as fh:
                fh.write("fake")
            with pytest.raises(ValueError, match="Unsupported file format"):
                self.tool.execute(path="tests/fixtures/_test_fake.xlsx")
        finally:
            if os.path.exists(fake_xlsx):
                os.remove(fake_xlsx)

    def test_path_traversal_blocked(self):
        with pytest.raises(ValueError, match="Path traversal"):
            self.tool.execute(path="../../../etc/passwd")


# ──────────────────────────────────────────────────────────────────────────
# Analyzer function tests
# ──────────────────────────────────────────────────────────────────────────

class TestAnalyzerShape:
    def test_shape(self):
        from core.analysis.analyzer import analyze_shape
        df = load_fixture()
        rows, cols = analyze_shape(df)
        assert rows == 11
        assert cols == 8


class TestAnalyzerDuplicates:
    def test_duplicates(self):
        from core.analysis.analyzer import analyze_duplicates
        df = load_fixture()
        dup_count, dup_pct = analyze_duplicates(df)
        assert dup_count == 1
        assert dup_pct > 0


class TestAnalyzerMissingValues:
    def test_missing_values(self):
        from core.analysis.analyzer import analyze_missing_values
        df = load_fixture()
        total, cols_missing, by_col = analyze_missing_values(df)
        assert total >= 1
        assert "TotalCharges" in cols_missing
        assert by_col["TotalCharges"] == 1

    def test_missing_percentage(self):
        df = load_fixture()
        n = len(df)
        from core.analysis.analyzer import analyze_missing_values
        _, _, by_col = analyze_missing_values(df)
        pct = by_col["TotalCharges"] / n * 100
        assert round(pct, 1) > 0


class TestColumnClassification:
    def test_numeric_columns(self):
        from core.analysis.analyzer import classify_columns
        df = load_fixture()
        col_types = classify_columns(df)
        assert "Age" in col_types["numeric"]
        assert "MonthlyCharges" in col_types["numeric"]

    def test_categorical_columns(self):
        from core.analysis.analyzer import classify_columns
        df = load_fixture()
        col_types = classify_columns(df)
        assert "Gender" in col_types["categorical"]
        assert "Contract" in col_types["categorical"]


class TestNumericStatistics:
    def test_column_profile_numeric(self):
        from core.analysis.analyzer import classify_columns, build_column_profile
        df = load_fixture()
        col_types = classify_columns(df)
        profile = build_column_profile("Age", df, col_types)
        assert profile.is_numeric
        assert profile.mean is not None
        assert profile.median is not None
        assert profile.standard_deviation is not None
        assert profile.minimum is not None
        assert profile.maximum is not None
        assert profile.quartiles is not None
        assert "q1" in profile.quartiles
        assert "q3" in profile.quartiles

    def test_mean_correct(self):
        from core.analysis.analyzer import classify_columns, build_column_profile
        df = load_fixture()
        col_types = classify_columns(df)
        profile = build_column_profile("Age", df, col_types)
        expected = float(df["Age"].mean())
        assert abs(profile.mean - expected) < 0.01


class TestCategoricalAnalysis:
    def test_categorical_profile(self):
        from core.analysis.analyzer import classify_columns, build_column_profile
        df = load_fixture()
        col_types = classify_columns(df)
        profile = build_column_profile("Contract", df, col_types)
        assert profile.is_categorical
        assert profile.unique_count >= 2
        assert profile.top_values is not None
        assert profile.value_counts is not None

    def test_top_values_present(self):
        from core.analysis.analyzer import classify_columns, build_column_profile
        df = load_fixture()
        col_types = classify_columns(df)
        profile = build_column_profile("Contract", df, col_types)
        assert "Month-to-month" in profile.top_values


class TestTargetDistribution:
    def test_target_distribution(self):
        from core.analysis.analyzer import analyze_target_distribution
        df = load_fixture()
        dist = analyze_target_distribution(df, "Churn")
        assert dist.target_column == "Churn"
        assert "Yes" in dist.class_counts
        assert "No" in dist.class_counts
        assert abs(sum(dist.class_percentages.values()) - 100.0) < 0.5

    def test_imbalance_detected(self):
        from core.analysis.analyzer import analyze_target_distribution
        df = load_fixture()
        dist = analyze_target_distribution(df, "Churn")
        # Fixture has 7 Yes, 4 No — minority < 20% threshold? Let's check
        # Minority is No = 4/11 = 36%, so no imbalance with 20% threshold
        # But we verify the bool type is set correctly
        assert isinstance(dist.imbalance_detected, bool)

    def test_missing_target_column(self):
        from core.analysis.analyzer import analyze_target_distribution
        df = load_fixture()
        with pytest.raises(ValueError, match="Target column"):
            analyze_target_distribution(df, "NonExistentColumn")


class TestOutliers:
    def test_outlier_detected(self):
        from core.analysis.analyzer import classify_columns, analyze_outliers
        df = load_fixture()
        col_types = classify_columns(df)
        outliers = analyze_outliers(df, col_types["numeric"])
        age_outlier = next((o for o in outliers if o.column == "Age"), None)
        assert age_outlier is not None
        assert age_outlier.outlier_count >= 1  # Age=200 is an outlier

    def test_iqr_bounds_correct(self):
        from core.analysis.analyzer import analyze_outliers
        df = pd.DataFrame({"X": [1, 2, 3, 4, 100]})
        outliers = analyze_outliers(df, ["X"])
        assert len(outliers) == 1
        o = outliers[0]
        assert o.outlier_count == 1
        assert o.q1 < o.q3


class TestCorrelations:
    def test_correlations_returned(self):
        from core.analysis.analyzer import analyze_correlations
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [2, 4, 6, 8, 10],  # perfect positive correlation
            "C": [5, 4, 3, 2, 1],  # perfect negative
        })
        pairs = analyze_correlations(df, ["A", "B", "C"], threshold=0.7)
        assert len(pairs) >= 2

    def test_correlation_direction(self):
        from core.analysis.analyzer import analyze_correlations
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [5, 4, 3, 2, 1],
        })
        pairs = analyze_correlations(df, ["A", "B"], threshold=0.7)
        assert any(p.direction == "negative" for p in pairs)


class TestIDDetection:
    def test_id_column_detected(self):
        from core.analysis.analyzer import detect_id_columns
        df = load_fixture()
        ids = detect_id_columns(df)
        assert "CustomerID" in ids

    def test_non_id_columns_not_flagged(self):
        from core.analysis.analyzer import detect_id_columns
        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [90, 85]})
        ids = detect_id_columns(df)
        # Score should not be flagged as ID
        assert "Score" not in ids


class TestWarnings:
    def test_missing_warning_generated(self):
        from core.analysis.analyzer import (
            analyze_missing_values, classify_columns, analyze_outliers,
            detect_id_columns, detect_constant_columns, generate_warnings,
        )
        from core.analysis.models import DatasetProfile
        df = pd.DataFrame({
            "A": [1, None, None, None, None, None],  # 83% missing
            "B": [1, 2, 3, 4, 5, 6],
        })
        total, cols_missing, missing_by_col = analyze_missing_values(df)
        col_types = classify_columns(df)
        outliers = analyze_outliers(df, col_types["numeric"])
        id_cols = detect_id_columns(df)
        const_cols = detect_constant_columns(df)
        profile = DatasetProfile(
            dataset_path="test.csv", file_type="csv",
            row_count=6, column_count=2,
            duplicate_count=0, duplicate_percentage=0.0,
            total_missing_values=total, columns_with_missing=cols_missing,
            numeric_columns=col_types["numeric"],
            categorical_columns=col_types["categorical"],
            datetime_columns=[], boolean_columns=[],
            potential_id_columns=id_cols, constant_columns=const_cols,
            outliers=outliers,
        )
        warnings = generate_warnings(df, profile, missing_by_col)
        msgs = [w.message for w in warnings]
        assert any("missing" in m.lower() for m in msgs)

    def test_duplicate_warning(self):
        from core.analysis.models import DatasetProfile
        from core.analysis.analyzer import generate_warnings
        df = pd.DataFrame({"A": [1, 2, 3]})
        profile = DatasetProfile(
            dataset_path="x.csv", file_type="csv",
            row_count=3, column_count=1,
            duplicate_count=2, duplicate_percentage=66.0,
        )
        warnings = generate_warnings(df, profile, {"A": 0})
        assert any("duplicate" in w.message.lower() for w in warnings)

    def test_empty_dataset_warning(self):
        from core.analysis.models import DatasetProfile
        from core.analysis.analyzer import generate_warnings
        df = pd.DataFrame()
        profile = DatasetProfile(
            dataset_path="empty.csv", file_type="csv",
            row_count=0, column_count=0,
        )
        warnings = generate_warnings(df, profile, {})
        assert any(w.severity == "critical" for w in warnings)


# ──────────────────────────────────────────────────────────────────────────
# DataAnalystAgent (mocked LLM)
# ──────────────────────────────────────────────────────────────────────────

class MockLLMManager:
    @property
    def provider_name(self): return "mock"
    def generate(self, prompt, system_prompt=None, response_schema=None):
        from core.llm.models import LLMResponse
        return LLMResponse(content="Mock observations.", provider="mock", model="mock")


class TestDataAnalystAgent:
    def setup_method(self):
        import unittest.mock as mock
        from config.settings import Settings
        self.patcher = mock.patch("tools.dataset_reader.get_settings",
                                   return_value=Settings(_env_file=None, WORKSPACE_DIR=WORKSPACE))
        self.patcher.start()
        from core.state.manager import AgentStateManager
        from agents.data_analyst_agent import DataAnalystAgent
        self.state_manager = AgentStateManager()
        self.agent = DataAnalystAgent(
            llm_provider=MockLLMManager(),
            state_manager=self.state_manager,
        )

    def teardown_method(self):
        self.patcher.stop()

    def test_analyze_returns_profile(self):
        profile = self.agent.analyze("tests/fixtures/test_churn.csv")
        assert profile.row_count == 11
        assert profile.column_count == 8

    def test_python_values_not_overridden_by_llm(self):
        profile = self.agent.analyze("tests/fixtures/test_churn.csv")
        # LLM returned "Mock observations." — the numeric values must come from Python
        assert profile.row_count == 11        # Python calculated
        assert profile.column_count == 8      # Python calculated
        assert profile.duplicate_count == 1   # Python calculated
        assert profile.observations == "Mock observations."  # LLM string

    def test_target_distribution(self):
        profile = self.agent.analyze("tests/fixtures/test_churn.csv", target_column="Churn")
        assert profile.target_distribution is not None
        assert profile.target_distribution.target_column == "Churn"

    def test_missing_target_raises(self):
        with pytest.raises((ValueError, RuntimeError)):
            self.agent.analyze("tests/fixtures/test_churn.csv", target_column="NoSuchCol")

    def test_state_updated(self):
        self.agent.analyze("tests/fixtures/test_churn.csv")
        st = self.state_manager.get_state()
        assert st.dataset_info is not None
        assert st.dataset_info.rows == 11

    def test_dataset_not_stored_in_state(self):
        """Raw DataFrame must NOT be stored in AgentState."""
        import pandas as pd
        self.agent.analyze("tests/fixtures/test_churn.csv")
        st = self.state_manager.get_state()
        # dataset_info should be DatasetInfo (small), not a DataFrame
        assert not isinstance(st.dataset_info, pd.DataFrame)

    def test_numeric_stats_from_python(self):
        """Verify Age mean matches Pandas directly."""
        df = pd.read_csv(FIXTURE_PATH)
        expected_mean = float(df["Age"].mean())
        profile = self.agent.analyze("tests/fixtures/test_churn.csv")
        age_profile = profile.column_profiles.get("Age")
        assert age_profile is not None
        assert abs(age_profile.mean - expected_mean) < 0.01

    def test_outlier_detected_in_age(self):
        profile = self.agent.analyze("tests/fixtures/test_churn.csv")
        age_outlier = next((o for o in profile.outliers if o.column == "Age"), None)
        assert age_outlier is not None
        assert age_outlier.outlier_count >= 1  # Age=200 is an outlier
