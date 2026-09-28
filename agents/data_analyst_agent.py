"""
Data Analyst Agent — Phase 6

Responsibilities:
  - Accept a dataset path + optional target column
  - Use deterministic Python/Pandas analysis (via core.analysis.analyzer)
  - Call LLM ONLY to interpret verified statistics
  - Return a structured DatasetProfile stored in AgentState
  
The LLM never calculates numbers. Python does.
"""
import json
import os
from typing import Optional

from agents.base_agent import BaseAgent
from agents.handoff import WorkerResult
from core.analysis.analyzer import (
    analyze_shape, analyze_duplicates, classify_columns,
    analyze_missing_values, build_column_profile,
    analyze_target_distribution, analyze_outliers, analyze_correlations,
    detect_id_columns, detect_constant_columns, generate_warnings,
    CORRELATION_THRESHOLD,
)
from core.analysis.models import DatasetProfile
from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager
from tools.dataset_reader import DatasetReaderTool

# Max sample rows sent to LLM context
MAX_SAMPLE_ROWS = 10
_SYSTEM_PROMPT = """\
You are the Data Analyst Agent in an MLForge multi-agent ML engineering system.

RULES:
1. Use ONLY the Python-verified statistics provided to you as source of truth.
2. Never invent, estimate, or recalculate dataset statistics yourself.
3. Do NOT modify the dataset or suggest dataset modifications in this step.
4. Identify data-quality issues based on the provided profile.
5. Explain important observations clearly and concisely.
6. Do not make unsupported causal claims.
7. Distinguish observations (what IS) from recommendations (what to DO).
8. Keep the response structured and concise. Use bullet points where appropriate.
9. If target imbalance is present, mention it clearly.
10. Your response will be stored as 'observations' in the DatasetProfile.
"""


class DataAnalystAgent(BaseAgent):
    def __init__(
        self,
        llm_provider: LLMManager,
        state_manager: AgentStateManager,
    ):
        super().__init__(
            llm_provider=llm_provider,
            name="data_analyst",
            role="data_analyst",
            system_prompt=_SYSTEM_PROMPT,
            state=state_manager,
        )
        self.state_manager = state_manager
        self._reader = DatasetReaderTool()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        dataset_path: str,
        target_column: Optional[str] = None,
        correlation_threshold: float = CORRELATION_THRESHOLD,
    ) -> DatasetProfile:
        """
        Full analysis pipeline. Returns a structured DatasetProfile.
        Numbers come from Python. Observations come from the LLM.
        """
        print(f"[DataAgent] Loading dataset: {dataset_path}")

        # Step 1: Load dataset
        try:
            df = self._reader.execute(path=dataset_path)
        except Exception as e:
            raise RuntimeError(f"DatasetReader error: {e}")

        ext = os.path.splitext(dataset_path)[1].lower().lstrip(".")
        print(f"[DataAgent] Dataset loaded: {len(df)} rows x {len(df.columns)} cols")

        # Step 2-4: Shape, Duplicates
        print("[DataAgent] Analyzing schema...")
        row_count, col_count = analyze_shape(df)

        print("[DataAgent] Analyzing duplicates...")
        dup_count, dup_pct = analyze_duplicates(df)

        # Step 5: Classify columns
        print("[DataAgent] Classifying columns...")
        col_types = classify_columns(df)

        # Step 6: Missing values
        print("[DataAgent] Analyzing missing values...")
        total_missing, cols_with_missing, missing_by_col = analyze_missing_values(df)

        # Step 7: Build column profiles
        column_profiles = {}
        for col in df.columns:
            column_profiles[col] = build_column_profile(col, df, col_types)

        # Step 8: Target validation & distribution
        target_dist = None
        if target_column:
            if target_column not in df.columns:
                raise ValueError(
                    f"TargetColumnNotFound: '{target_column}' does not exist in dataset columns: {list(df.columns)}"
                )
            print("[DataAgent] Analyzing target distribution...")
            target_dist = analyze_target_distribution(df, target_column)

        # Step 9: Numeric statistics already embedded in column profiles

        # Step 10: Outlier detection
        print("[DataAgent] Detecting outliers...")
        outliers = analyze_outliers(df, col_types["numeric"])

        # Step 11: Correlations
        print("[DataAgent] Calculating correlations...")
        strong_corrs = analyze_correlations(df, col_types["numeric"], correlation_threshold)

        # Step 12: ID + constant column detection
        id_cols = detect_id_columns(df)
        const_cols = detect_constant_columns(df)

        # Step 13: Assemble DatasetProfile (numbers only, no LLM yet)
        print("[DataAgent] Building DatasetProfile...")
        profile = DatasetProfile(
            dataset_path=dataset_path,
            file_type=ext or "csv",
            row_count=row_count,
            column_count=col_count,
            columns=list(df.columns),
            data_types={col: str(df[col].dtype) for col in df.columns},
            column_profiles=column_profiles,
            total_missing_values=total_missing,
            columns_with_missing=cols_with_missing,
            duplicate_count=dup_count,
            duplicate_percentage=dup_pct,
            numeric_columns=col_types["numeric"],
            categorical_columns=col_types["categorical"],
            datetime_columns=col_types["datetime"],
            boolean_columns=col_types["boolean"],
            target_column=target_column,
            target_distribution=target_dist,
            outliers=outliers,
            strong_correlations=strong_corrs,
            correlation_threshold=correlation_threshold,
            potential_id_columns=id_cols,
            constant_columns=const_cols,
        )

        # Step 14: Generate warnings (uses completed profile)
        profile.warnings = generate_warnings(df, profile, missing_by_col)

        # Step 15: LLM interpretation (only if not empty dataset)
        if row_count > 0:
            print("[DataAgent] Generating observations...")
            profile.observations = self._generate_observations(profile)

        # Step 16: Store in AgentState
        self._store_in_state(profile)

        return profile

    def run(self, user_input: str) -> str:
        """BaseAgent compatibility — parse path from user input."""
        parts = user_input.strip().split()
        path = parts[0] if parts else ""
        target = parts[1] if len(parts) > 1 else None
        try:
            profile = self.analyze(dataset_path=path, target_column=target)
            return self._format_summary(profile)
        except Exception as e:
            return f"[DataAnalystAgent] Error: {e}"

    def execute_task_from_step(self, step_info: dict) -> WorkerResult:
        """
        Called by the Executor when agent_role == 'data_analyst'.
        Expects step_info keys: dataset_path, target_column (optional).
        """
        import uuid
        task_id = step_info.get("step_id", str(uuid.uuid4()))
        dataset_path = step_info.get("dataset_path") or step_info.get("input_requirements", {}).get("dataset_path")
        target_column = step_info.get("target_column") or step_info.get("input_requirements", {}).get("target_column")

        if not dataset_path:
            return WorkerResult(
                task_id=task_id, agent="data_analyst",
                status="failed", result="",
                errors=["dataset_path not provided in step info"],
            )

        try:
            profile = self.analyze(dataset_path=dataset_path, target_column=target_column)
            summary = self._format_summary(profile)
            return WorkerResult(
                task_id=task_id, agent="data_analyst",
                status="completed", result=summary,
            )
        except Exception as e:
            return WorkerResult(
                task_id=task_id, agent="data_analyst",
                status="failed", result="", errors=[str(e)],
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_observations(self, profile: DatasetProfile) -> str:
        """Send only the structured summary to LLM — never raw data rows."""
        summary_lines = [
            f"Dataset: {os.path.basename(profile.dataset_path)}",
            f"Shape: {profile.row_count} rows x {profile.column_count} columns",
            f"Numeric columns: {len(profile.numeric_columns)} — {profile.numeric_columns}",
            f"Categorical columns: {len(profile.categorical_columns)} — {profile.categorical_columns}",
            f"Datetime columns: {len(profile.datetime_columns)}",
            f"Total missing values: {profile.total_missing_values}",
            f"Columns with missing: {profile.columns_with_missing}",
            f"Duplicate rows: {profile.duplicate_count} ({profile.duplicate_percentage}%)",
        ]

        if profile.target_distribution:
            td = profile.target_distribution
            dist_str = ", ".join(f"{k}: {v}%" for k, v in td.class_percentages.items())
            summary_lines += [
                f"Target column: {td.target_column}",
                f"Target distribution: {dist_str}",
                f"Class imbalance detected: {td.imbalance_detected}",
            ]

        if profile.outliers:
            sig = [o for o in profile.outliers if o.outlier_count > 0]
            summary_lines.append(f"Columns with outliers: {[o.column for o in sig]}")

        if profile.strong_correlations:
            corr_str = "; ".join(
                f"{c.column_a} <-> {c.column_b} = {c.correlation}"
                for c in profile.strong_correlations[:5]
            )
            summary_lines.append(f"Strong correlations (>={profile.correlation_threshold}): {corr_str}")

        if profile.constant_columns:
            summary_lines.append(f"Constant columns: {profile.constant_columns}")

        if profile.potential_id_columns:
            summary_lines.append(f"Potential ID columns: {profile.potential_id_columns}")

        warning_strs = [f"[{w.severity.upper()}] {w.message}" for w in profile.warnings]
        if warning_strs:
            summary_lines.append("Warnings:\n" + "\n".join(warning_strs))

        summary_text = "\n".join(summary_lines)
        prompt = (
            f"The following dataset profile was computed by Python/Pandas tools.\n\n"
            f"{summary_text}\n\n"
            f"Based ONLY on the above verified statistics, provide:\n"
            f"1. Key observations about this dataset (3-6 bullet points)\n"
            f"2. Recommended next steps for ML preprocessing (2-4 bullet points)\n"
            f"Be concise and structured."
        )

        try:
            resp = self.llm_provider.generate(prompt=prompt, system_prompt=self.system_prompt)
            return resp.content
        except Exception as e:
            return f"[LLM observations unavailable: {e}]"

    def _store_in_state(self, profile: DatasetProfile):
        """Store profile summary in AgentState.dataset_info (no raw DataFrame)."""
        from core.state.models import DatasetInfo
        info = DatasetInfo(
            name=os.path.basename(profile.dataset_path),
            path=profile.dataset_path,
            rows=profile.row_count,
            columns=profile.column_count,
            target_column=profile.target_column,
        )
        self.state_manager.set_dataset_info(info)
        # Also store the full profile as a dict in workflow_result for downstream agents
        self.state_manager.state.workflow_result = {
            "type": "dataset_profile",
            "data": profile.model_dump(exclude={"column_profiles"}),
        }
        self.state_manager.add_decision(
            action="dataset_analyzed",
            reason=f"DatasetProfile built for {profile.dataset_path}",
        )

    def _format_summary(self, profile: DatasetProfile) -> str:
        lines = [
            "=" * 41,
            "DATASET PROFILE",
            "=" * 41,
            f"Dataset:    {os.path.basename(profile.dataset_path)}",
            f"Rows:       {profile.row_count}",
            f"Columns:    {profile.column_count}",
            f"Missing:    {profile.total_missing_values}",
            f"Duplicates: {profile.duplicate_count} ({profile.duplicate_percentage}%)",
            "",
            f"Numeric:     {len(profile.numeric_columns)} — {profile.numeric_columns}",
            f"Categorical: {len(profile.categorical_columns)} — {profile.categorical_columns}",
        ]

        if profile.target_distribution:
            td = profile.target_distribution
            dist_str = "  ".join(f"{k}: {v}%" for k, v in td.class_percentages.items())
            lines += [
                "",
                f"Target: {td.target_column}",
                f"Distribution: {dist_str}",
                f"Imbalance: {'Yes' if td.imbalance_detected else 'No'}",
            ]

        if profile.outliers:
            sig = [o for o in profile.outliers if o.outlier_count > 0]
            if sig:
                lines.append(f"\nOutliers detected in: {[o.column for o in sig]}")

        if profile.strong_correlations:
            lines.append("\nStrong Correlations:")
            for c in profile.strong_correlations[:5]:
                lines.append(f"  {c.column_a} <-> {c.column_b} = {c.correlation}")

        if profile.warnings:
            lines.append("\nWarnings:")
            for w in profile.warnings:
                lines.append(f"  [{w.severity.upper()}] {w.message}")

        if profile.observations:
            lines += ["", "Observations:", profile.observations]

        lines.append("=" * 41)
        return "\n".join(lines)
