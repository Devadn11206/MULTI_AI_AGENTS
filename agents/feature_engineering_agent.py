"""
Feature Engineering Agent — Phase 7

Responsibilities:
- Receive DatasetProfile from Phase 6
- Call LLMManager to generate a FeatureEngineeringPlan
- Validate and reject unsafe operations
- Execute the plan via FeatureTransformer (Python/Pandas only)
- Save processed dataset to data/processed/
- Store FeatureEngineeringResult in AgentState
- Return WorkerResult to Orchestrator

The LLM plans. Python executes. The target is always protected.
"""
import json
import os
import uuid
from typing import Optional

import pandas as pd

from agents.base_agent import BaseAgent
from agents.handoff import WorkerResult
from config.settings import get_settings
from core.analysis.models import DatasetProfile
from core.feature_engineering.models import (
    FeatureEngineeringPlan, FeatureEngineeringResult, TransformationOperation,
)
from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager
from tools.dataset_reader import DatasetReaderTool
from tools.feature_transformer import FeatureTransformer, SUPPORTED_OPERATIONS

_SYSTEM_PROMPT = """\
You are the Feature Engineering Agent in MLForge, a multi-agent ML engineering system.

RULES:
1. Use DatasetProfile as the ONLY source of truth about the dataset. Do NOT invent column names.
2. Never generate arbitrary executable Python code.
3. Never modify the dataset directly — only produce a structured plan.
4. Always protect the target column. Never include it in transformations.
5. Avoid target leakage — do not derive features directly from the target column.
6. Prefer simple, explainable transformations over complex ones.
7. Do not remove features without a clear reason.
8. Do not train models or evaluate models.
9. Flag any uncertain recommendations in the 'warnings' field.
10. Produce a valid JSON matching the FeatureEngineeringPlan schema EXACTLY.

SUPPORTED OPERATIONS (use only these):
- impute_numeric: strategy = "median" | "mean" | "constant"
- impute_categorical: strategy = "mode" | "constant"
- one_hot_encode: columns to one-hot-encode (not the target)
- binary_encode: for binary categorical columns
- scale: strategy = "standard" | "minmax"
- remove_column: columns to remove
- create_ratio: numerator, denominator, new_column
- create_difference: columns[0] - columns[1], new_column
- create_sum: columns list, new_column
- create_log: columns to log-transform, new_column
- extract_datetime_features: datetime columns

Return ONLY valid JSON with this schema:
{
  "target_column": "<target>",
  "operations": [
    {"operation": "...", "columns": [...], "strategy": "...", "reason": "..."}
  ],
  "feature_removal": ["col1", "col2"],
  "duplicate_handling": "keep",
  "reasoning": "...",
  "warnings": []
}
"""


class FeatureEngineeringAgent(BaseAgent):
    """
    First specialized transformation agent in MLForge.
    Accepts a DatasetProfile, generates a plan via LLM, executes via FeatureTransformer.
    """

    PROCESSED_DIR = "data/processed"

    def __init__(
        self,
        llm_provider: LLMManager,
        state_manager: AgentStateManager,
        workspace_dir: Optional[str] = None,
    ):
        super().__init__(
            llm_provider=llm_provider,
            name="feature_engineer",
            role="feature_engineer",
            system_prompt=_SYSTEM_PROMPT,
            state=state_manager,
        )
        self.state_manager = state_manager
        self._reader = DatasetReaderTool()
        # Allow explicit injection for tests; fall back to settings
        if workspace_dir is not None:
            self._workspace = os.path.abspath(workspace_dir)
        else:
            try:
                settings = get_settings()
                self._workspace = os.path.abspath(
                    getattr(settings, "WORKSPACE_DIR", os.getcwd())
                )
            except Exception:
                self._workspace = os.getcwd()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def engineer(
        self,
        dataset_path: str,
        target_column: str,
        dataset_profile: Optional[DatasetProfile] = None,
        ml_problem: Optional[str] = None,
    ) -> FeatureEngineeringResult:
        """
        Full feature engineering pipeline.
        Returns FeatureEngineeringResult (no raw DataFrame stored in AgentState).
        """
        print(f"[FEAgent] Starting feature engineering for: {dataset_path}")

        # Step 1: Load dataset
        try:
            df = self._reader.execute(path=dataset_path)
        except Exception as e:
            return self._error_result(f"Failed to load dataset: {e}", target_column)

        original_shape = list(df.shape)
        original_cols = list(df.columns)
        missing_before = int(df.isna().sum().sum())
        dups_before = int(df.duplicated().sum())

        # Step 2: Validate target
        if target_column not in df.columns:
            return self._error_result(
                f"Target column '{target_column}' not found in dataset.", target_column
            )

        # Step 3: Build profile summary for LLM (no raw rows)
        profile_summary = self._build_profile_summary(df, dataset_profile, target_column, ml_problem)

        # Step 4: LLM generates FeatureEngineeringPlan
        print("[FEAgent] Generating Feature Engineering Plan (LLM)...")
        plan = self._generate_plan(profile_summary, target_column)
        if plan is None:
            return self._error_result("LLM failed to produce a valid plan.", target_column)

        # Step 5: Validate and sanitize the plan
        plan, plan_warnings = self._validate_plan(plan, df)

        # Step 6: Execute via FeatureTransformer (Python only)
        print("[FEAgent] Executing transformations...")
        transformer = FeatureTransformer(target_column=target_column)
        try:
            transformed_df, records = transformer.fit_transform(df, plan)
        except Exception as e:
            return self._error_result(f"Transformation failed: {e}", target_column)

        # Step 7: Post-transformation validation
        validation_warnings = self._validate_output(
            df, transformed_df, target_column, plan
        )
        all_warnings = plan_warnings + validation_warnings

        # Step 8: Save processed dataset
        processed_path = self._save_processed(transformed_df, dataset_path)

        # Step 9: Build result
        added = [c for c in transformed_df.columns if c not in original_cols]
        removed = [c for c in original_cols if c not in transformed_df.columns]
        missing_after = int(transformed_df.isna().sum().sum())
        dups_after = int(transformed_df.duplicated().sum())
        rows_removed = dups_before - dups_after if plan.duplicate_handling == "remove" else 0

        result = FeatureEngineeringResult(
            status="completed",
            original_shape=original_shape,
            transformed_shape=list(transformed_df.shape),
            original_columns=original_cols,
            transformed_columns=list(transformed_df.columns),
            added_features=added,
            removed_features=removed,
            transformations=records,
            warnings=all_warnings,
            plan=plan,
            target_column=target_column,
            processed_dataset_path=processed_path,
            missing_before=missing_before,
            missing_after=missing_after,
            duplicates_before=dups_before,
            duplicates_after=dups_after,
            rows_removed_duplicates=rows_removed,
        )

        # Step 10: Update AgentState
        self._update_state(result)

        print("[FEAgent] Feature engineering complete.")
        return result

    def run(self, user_input: str) -> str:
        """BaseAgent compatibility."""
        parts = user_input.strip().split()
        path = parts[0] if parts else ""
        target = parts[1] if len(parts) > 1 else ""
        try:
            result = self.engineer(dataset_path=path, target_column=target)
            return self._format_summary(result)
        except Exception as e:
            return f"[FeatureEngineeringAgent] Error: {e}"

    def execute_task_from_step(self, step_info: dict) -> WorkerResult:
        """Called by Executor when agent_role == 'feature_engineer'."""
        task_id = step_info.get("step_id", str(uuid.uuid4()))
        dataset_path = (
            step_info.get("dataset_path")
            or step_info.get("input_requirements", {}).get("dataset_path")
        )
        target_column = (
            step_info.get("target_column")
            or step_info.get("input_requirements", {}).get("target_column", "")
        )
        # Get DatasetProfile from state if available
        dataset_profile = None
        state = self.state_manager.get_state()
        if state.workflow_result and state.workflow_result.get("type") == "dataset_profile":
            try:
                profile_data = state.workflow_result.get("data", {})
                dataset_profile = DatasetProfile(**profile_data)
            except Exception:
                pass

        if not dataset_path:
            return WorkerResult(
                task_id=task_id, agent="feature_engineer",
                status="failed", result="",
                errors=["dataset_path not provided"],
            )

        try:
            result = self.engineer(
                dataset_path=dataset_path,
                target_column=target_column,
                dataset_profile=dataset_profile,
            )
            summary = self._format_summary(result)
            return WorkerResult(
                task_id=task_id, agent="feature_engineer",
                status="completed" if result.status == "completed" else "failed",
                result=summary,
                errors=result.errors,
            )
        except Exception as e:
            return WorkerResult(
                task_id=task_id, agent="feature_engineer",
                status="failed", result="", errors=[str(e)],
            )

    # ------------------------------------------------------------------
    # LLM plan generation
    # ------------------------------------------------------------------

    def _generate_plan(self, profile_summary: str, target_column: str) -> Optional[FeatureEngineeringPlan]:
        prompt = (
            f"Based on the following dataset profile, generate a FeatureEngineeringPlan.\n\n"
            f"{profile_summary}\n\n"
            f"Target column: {target_column}\n\n"
            f"Return ONLY valid JSON matching the FeatureEngineeringPlan schema. "
            f"Do not add markdown, code fences, or explanatory text outside the JSON."
        )
        resp = None
        try:
            resp = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                response_schema=FeatureEngineeringPlan,
            )
            # Try parsing directly
            plan = FeatureEngineeringPlan.model_validate_json(resp.content)
            return plan
        except Exception:
            # Fallback: try extracting JSON from response
            try:
                raw = resp.content if resp else ""
                # Strip markdown fences if present
                raw = raw.strip().strip("```json").strip("```").strip()
                plan = FeatureEngineeringPlan.model_validate_json(raw)
                return plan
            except Exception as e2:
                print(f"[FEAgent] Failed to parse LLM plan: {e2}")
                return None

    # ------------------------------------------------------------------
    # Profile summary for LLM (never sends raw rows)
    # ------------------------------------------------------------------

    def _build_profile_summary(
        self,
        df: pd.DataFrame,
        profile: Optional[DatasetProfile],
        target_column: str,
        ml_problem: Optional[str],
    ) -> str:
        lines = []
        if ml_problem:
            lines.append(f"ML Problem: {ml_problem}")
        lines += [
            f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns",
            f"Target column: {target_column}",
            f"Columns: {list(df.columns)}",
        ]

        # Use DatasetProfile if available
        if profile:
            lines += [
                f"Numeric columns: {profile.numeric_columns}",
                f"Categorical columns: {profile.categorical_columns}",
                f"Missing values total: {profile.total_missing_values}",
                f"Columns with missing: {profile.columns_with_missing}",
                f"Duplicate rows: {profile.duplicate_count}",
                f"Potential ID columns: {profile.potential_id_columns}",
                f"Constant columns: {profile.constant_columns}",
            ]
            if profile.warnings:
                warn_msgs = [f"  [{w.severity.upper()}] {w.message}" for w in profile.warnings]
                lines.append("Data quality warnings:\n" + "\n".join(warn_msgs))
        else:
            # Infer basic info from DataFrame directly
            numeric = df.select_dtypes(include="number").columns.tolist()
            categorical = df.select_dtypes(include="object").columns.tolist()
            missing = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
            lines += [
                f"Numeric columns: {numeric}",
                f"Categorical columns: {categorical}",
                f"Missing values per column: {missing}",
            ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Plan validation — reject unsafe operations
    # ------------------------------------------------------------------

    def _validate_plan(
        self, plan: FeatureEngineeringPlan, df: pd.DataFrame
    ):
        """Remove unsafe/invalid operations. Return sanitized plan + warnings."""
        warnings_out = []
        safe_ops = []

        for op in plan.operations:
            # Check supported
            if op.operation not in SUPPORTED_OPERATIONS:
                warnings_out.append(
                    f"Unsupported operation '{op.operation}' was rejected."
                )
                continue
            # Check target not in columns
            if self.target_column_in_op(op, plan.target_column):
                warnings_out.append(
                    f"Operation '{op.operation}' on target column rejected — target is protected."
                )
                continue
            # Check columns exist
            bad_cols = [c for c in op.columns if c not in df.columns]
            if bad_cols and op.operation not in {"create_ratio", "create_difference"}:
                warnings_out.append(
                    f"Operation '{op.operation}' references unknown columns {bad_cols} — skipped."
                )
                continue
            safe_ops.append(op)

        # Remove target from feature_removal
        if plan.target_column in plan.feature_removal:
            plan.feature_removal.remove(plan.target_column)
            warnings_out.append(f"Target column '{plan.target_column}' was removed from feature_removal list.")

        # Leakage check — warn if any new_column name contains the target name
        for op in safe_ops:
            nc = op.new_column or ""
            if plan.target_column.lower() in nc.lower():
                warnings_out.append(
                    f"Potential target leakage: new column '{nc}' contains target name."
                )

        plan.operations = safe_ops
        return plan, warnings_out

    @staticmethod
    def target_column_in_op(op: TransformationOperation, target: str) -> bool:
        if target in op.columns:
            return True
        if op.numerator == target or op.denominator == target:
            return True
        return False

    # ------------------------------------------------------------------
    # Post-transformation validation
    # ------------------------------------------------------------------

    def _validate_output(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        target_column: str,
        plan: FeatureEngineeringPlan,
    ) -> list:
        warnings_out = []

        # Target must still exist
        if target_column not in transformed_df.columns:
            warnings_out.append(f"CRITICAL: Target column '{target_column}' was lost during transformation!")

        # Target must not have changed
        if target_column in original_df.columns and target_column in transformed_df.columns:
            if not original_df[target_column].equals(transformed_df[target_column]):
                warnings_out.append(f"CRITICAL: Target column '{target_column}' values were modified!")

        # Row count should only change if duplicates were removed
        if len(transformed_df) != len(original_df) and plan.duplicate_handling != "remove":
            warnings_out.append(
                f"Row count changed unexpectedly: {len(original_df)} -> {len(transformed_df)}"
            )

        # Any new unexpected NaNs (compared to handled columns)
        new_missing = int(transformed_df.isna().sum().sum())
        old_missing = int(original_df.isna().sum().sum())
        if new_missing > old_missing:
            warnings_out.append(
                f"Missing values INCREASED after transformation: {old_missing} -> {new_missing}"
            )

        return warnings_out

    # ------------------------------------------------------------------
    # Save processed dataset
    # ------------------------------------------------------------------

    def _save_processed(self, df: pd.DataFrame, original_path: str) -> Optional[str]:
        try:
            processed_dir = os.path.join(self._workspace, self.PROCESSED_DIR)
            os.makedirs(processed_dir, exist_ok=True)
            basename = os.path.splitext(os.path.basename(original_path))[0]
            out_filename = f"{basename}_features.csv"
            out_path = os.path.join(processed_dir, out_filename)
            df.to_csv(out_path, index=False)
            rel_path = os.path.join(self.PROCESSED_DIR, out_filename).replace("\\", "/")
            print(f"[FEAgent] Saved processed dataset to: {rel_path}")
            return rel_path
        except Exception as e:
            print(f"[FEAgent] Warning: could not save processed dataset: {e}")
            return None

    # ------------------------------------------------------------------
    # AgentState update
    # ------------------------------------------------------------------

    def _update_state(self, result: FeatureEngineeringResult):
        state = self.state_manager.state
        # Store plan + result as dicts (no raw DataFrames)
        if not hasattr(state, "feature_engineering_plan"):
            state.__dict__["feature_engineering_plan"] = None
        if not hasattr(state, "feature_engineering_result"):
            state.__dict__["feature_engineering_result"] = None
        if not hasattr(state, "processed_dataset_reference"):
            state.__dict__["processed_dataset_reference"] = None

        state.feature_engineering_plan = result.plan.model_dump() if result.plan else None
        state.feature_engineering_result = result.model_dump(exclude={"plan"})
        state.processed_dataset_reference = result.processed_dataset_path

        self.state_manager.add_decision(
            action="feature_engineering_completed",
            reason=f"Transformed dataset saved to {result.processed_dataset_path}",
        )

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------

    def _error_result(self, message: str, target_column: str) -> FeatureEngineeringResult:
        print(f"[FEAgent] Error: {message}")
        return FeatureEngineeringResult(
            status="failed",
            original_shape=[0, 0],
            transformed_shape=[0, 0],
            target_column=target_column,
            errors=[message],
        )

    # ------------------------------------------------------------------
    # Summary formatting
    # ------------------------------------------------------------------

    def _format_summary(self, result: FeatureEngineeringResult) -> str:
        lines = [
            "=" * 41,
            "FEATURE ENGINEERING RESULT",
            "=" * 41,
            f"Status:        {result.status.upper()}",
            f"Original:      {result.original_shape[0]} rows x {result.original_shape[1]} cols",
            f"Transformed:   {result.transformed_shape[0]} rows x {result.transformed_shape[1]} cols",
            f"Added:         {len(result.added_features)} features — {result.added_features}",
            f"Removed:       {len(result.removed_features)} features — {result.removed_features}",
            f"Missing:       {result.missing_before} -> {result.missing_after}",
            f"Duplicates:    {result.duplicates_before} -> {result.duplicates_after}",
        ]
        if result.processed_dataset_path:
            lines.append(f"Output:        {result.processed_dataset_path}")
        if result.warnings:
            lines.append("\nWarnings:")
            for w in result.warnings:
                lines.append(f"  [WARN] {w}")
        if result.transformations:
            lines.append("\nTransformations Applied:")
            for t in result.transformations:
                status_sym = "[OK]" if t.status == "completed" else f"[{t.status.upper()}]"
                lines.append(f"  {status_sym} {t.operation} -> {t.output_columns or t.input_columns}")
        lines.append("=" * 41)
        return "\n".join(lines)
