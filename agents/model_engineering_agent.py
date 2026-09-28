import json
import uuid
from typing import Any, Dict, Optional, Tuple, List

from agents.base_agent import BaseAgent
from agents.handoff import WorkerResult
from core.analysis.models import DatasetProfile
from core.feature_engineering.models import FeatureEngineeringResult
from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager
from core.model_engineering.config import DEFAULT_RANDOM_STATE, MAX_CANDIDATE_MODELS
from core.model_engineering.registry import get_registry
from core.model_engineering.models import ExperimentPlan, ExperimentConstraints


_SYSTEM_PROMPT = """You are the Model Engineering Agent in MLForge.

Your objective is to design a structured ExperimentPlan for a machine learning problem.

RULES:
1. Analyze the ML problem carefully.
2. Use DatasetProfile as the source of truth for the raw data characteristics.
3. Use FeatureEngineeringResult as the source of truth for available features.
4. Select a small set of appropriate candidate models (MAX 3-5).
5. Always include a baseline when appropriate (e.g., Logistic Regression for classification, Linear Regression for regression).
6. Consider user constraints (e.g., metric priorities).
7. Consider class imbalance and recommend strategies (e.g., stratified CV, class weights) without actively modifying the dataset here.
8. Consider dataset size and feature characteristics for scaling requirements.
9. DO NOT invent performance metrics. DO NOT hallucinate accuracy. 
10. DO NOT train models. DO NOT execute arbitrary Python code. DO NOT modify datasets.
11. Return a structured ExperimentPlan in JSON matching the provided schema.
12. Explain the rationale for model selection clearly.
13. Prefer simple experiments before complex ones.

You will be provided with:
- ML Problem Description
- Task Type
- Target Column
- Dataset Profile (summary)
- Feature Engineering Result (summary)
- Available Models from the Registry
- Constraints

You MUST return a JSON object matching the `ExperimentPlan` schema exactly.
"""


class ModelEngineeringAgent(BaseAgent):
    """
    Model Engineering Agent (Phase 8).
    Selects ML algorithms and creates a structured experiment plan.
    Does NOT train models.
    """

    def __init__(
        self,
        llm_provider: LLMManager,
        state_manager: AgentStateManager,
    ):
        super().__init__(
            llm_provider=llm_provider,
            name="model_engineer",
            role="model_engineer",
            system_prompt=_SYSTEM_PROMPT,
            state=state_manager,
        )
        self.state_manager = state_manager
        self._registry = get_registry()

    def plan_experiment(
        self,
        ml_problem: str,
        target_column: str,
        dataset_profile: DatasetProfile,
        feature_engineering_result: FeatureEngineeringResult,
        constraints: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None,
    ) -> ExperimentPlan:
        """
        Determines the task type, selects models, and generates an ExperimentPlan.
        """
        print("[MEAgent] Starting model engineering phase...")

        # 1. Determine task type
        detected_task_type = self._determine_task_type(
            target_column=target_column,
            dataset_profile=dataset_profile,
            explicit_task_type=task_type
        )
        
        # 2. Get available models for this task type
        available_models = self._registry.list_models(task_type=detected_task_type)
        
        # We only pass models that are actually installed/available in the environment
        installed_models = [m for m in available_models if self._registry.is_available(m.model_id)]
        
        if not installed_models:
            raise ValueError(f"No installed models found for task type '{detected_task_type}'")

        models_json = [m.model_dump() for m in installed_models]

        # 3. Create context for LLM
        prompt = self._build_prompt(
            ml_problem=ml_problem,
            target_column=target_column,
            task_type=detected_task_type,
            dataset_profile=dataset_profile,
            feature_engineering_result=feature_engineering_result,
            available_models=models_json,
            constraints=constraints,
        )

        # 4. Ask LLM to generate ExperimentPlan
        print(f"[MEAgent] Generating experiment plan for '{detected_task_type}'...")
        response = self.llm_provider.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            response_schema=ExperimentPlan
        )

        try:
            plan_data = json.loads(response.content)
            
            # 5. Add unique ID if not provided
            if not plan_data.get("experiment_id"):
                plan_data["experiment_id"] = f"exp_{uuid.uuid4().hex[:8]}"
            
            # Enforce deterministic random_state in plan if not strictly defined
            if "random_state" not in plan_data or plan_data["random_state"] is None:
                plan_data["random_state"] = DEFAULT_RANDOM_STATE

            # Parse to Pydantic model
            plan = ExperimentPlan(**plan_data)
            
            # 6. Validate & filter candidates against max
            self._validate_and_filter_plan(plan, installed_models)

            # 7. Update AgentState
            st = self.state_manager.get_state()
            st.experiment_plan = plan.model_dump()
            st.model_candidates = [m.model_dump() for m in plan.candidate_models]
            st.selected_models = [m.model_id for m in plan.candidate_models]
            st.model_engineering_warnings = plan.warnings

            print(f"[MEAgent] Generated Experiment Plan with {len(plan.candidate_models)} candidates.")
            return plan

        except Exception as e:
            print(f"[MEAgent] Failed to parse Experiment Plan: {e}")
            raise

    def execute_task_from_step(self, step_info: Dict[str, Any]) -> WorkerResult:
        """
        Orchestrator integration.
        """
        task_id = step_info.get("step_id", "unknown")
        ml_problem = step_info.get("ml_problem", "Unknown problem")
        target_column = step_info.get("target_column")
        task_type = step_info.get("task_type")
        constraints = step_info.get("constraints")
        
        st = self.state_manager.get_state()
        
        # We expect DataAnalyst and FeatureEngineering to have populated state
        profile_data = None
        if st.workflow_result and st.workflow_result.get("type") == "dataset_profile":
            profile_data = st.workflow_result.get("data")
            
        fe_result_data = st.feature_engineering_result
        
        if not profile_data or not fe_result_data:
            return WorkerResult(
                task_id=task_id,
                agent=self.name,
                status="failed",
                result="",
                errors=["DatasetProfile or FeatureEngineeringResult missing from state."]
            )
            
        try:
            dataset_profile = DatasetProfile(**profile_data)
            fe_result = FeatureEngineeringResult(**fe_result_data)
            
            plan = self.plan_experiment(
                ml_problem=ml_problem,
                target_column=target_column,
                dataset_profile=dataset_profile,
                feature_engineering_result=fe_result,
                constraints=constraints,
                task_type=task_type
            )
            
            summary = self._format_summary(plan)
            
            return WorkerResult(
                task_id=task_id,
                agent=self.name,
                status="completed",
                result={
                    "type": "experiment_plan",
                    "data": plan.model_dump(),
                    "summary": summary
                }
            )
            
        except Exception as e:
            return WorkerResult(
                task_id=task_id,
                agent=self.name,
                status="failed",
                result="",
                errors=[str(e)]
            )

    def _determine_task_type(
        self, 
        target_column: str, 
        dataset_profile: DatasetProfile,
        explicit_task_type: Optional[str] = None
    ) -> str:
        """
        Determines if the task is binary_classification, multiclass_classification, or regression.
        """
        if explicit_task_type:
            # Respect explicit type
            return explicit_task_type
            
        # Find target in dataset profile
        if target_column in dataset_profile.categorical_columns or target_column in dataset_profile.boolean_columns:
            # It's classification. Check number of classes.
            if dataset_profile.target_distribution and dataset_profile.target_distribution.number_of_classes > 2:
                return "multiclass_classification"
            return "binary_classification"
        elif target_column in dataset_profile.numeric_columns:
            # It's numeric. Could be regression or encoded classification.
            if dataset_profile.target_distribution:
                n_classes = dataset_profile.target_distribution.number_of_classes
                if n_classes == 2:
                    return "binary_classification"
                elif n_classes < 20:
                    return "multiclass_classification"
            return "regression"
            
        print(f"[MEAgent] Warning: Target '{target_column}' not found in profile. Assuming regression.")
        return "regression"

    def _build_prompt(
        self,
        ml_problem: str,
        target_column: str,
        task_type: str,
        dataset_profile: DatasetProfile,
        feature_engineering_result: FeatureEngineeringResult,
        available_models: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]]
    ) -> str:
        
        # Extract lightweight summaries for prompt
        imbalance_flag = dataset_profile.target_distribution.imbalance_detected if dataset_profile.target_distribution else "Unknown"
        
        # Features after FE
        final_features = feature_engineering_result.transformed_columns
        num_features = feature_engineering_result.transformed_shape[1] - 1 # excluding target
        num_rows = feature_engineering_result.transformed_shape[0]
        
        prompt = f"""
ML Problem: {ml_problem}
Task Type Detected: {task_type}
Target Column: {target_column}

# Dataset Metadata (After Feature Engineering)
Rows: {num_rows}
Features: {num_features}
Class Imbalance Detected: {imbalance_flag}
Processed Dataset Path: {feature_engineering_result.processed_dataset_path}

# Available Installed Models for this Task Type:
{json.dumps(available_models, indent=2)}

# Constraints:
{json.dumps(constraints or {}, indent=2)}

Please generate an ExperimentPlan JSON object.
Select a maximum of {MAX_CANDIDATE_MODELS} candidate models that are best suited for this task.
Always include a baseline model.
Do NOT fabricate performance metrics.
"""
        return prompt
        
    def _validate_and_filter_plan(self, plan: ExperimentPlan, installed_models: List[Any]):
        """
        Validates the returned plan against installed models and max constraints.
        """
        installed_ids = {m.model_id for m in installed_models}
        
        valid_candidates = []
        for cand in plan.candidate_models:
            if cand.model_id in installed_ids:
                valid_candidates.append(cand)
            else:
                plan.warnings.append(f"Model '{cand.model_id}' is not available or not installed. Removing from candidates.")
                
        # Limit to max
        if len(valid_candidates) > MAX_CANDIDATE_MODELS:
            plan.warnings.append(f"LLM proposed {len(valid_candidates)} models. Limiting to top {MAX_CANDIDATE_MODELS}.")
            
            # Sort by priority and slice
            valid_candidates.sort(key=lambda x: x.priority)
            valid_candidates = valid_candidates[:MAX_CANDIDATE_MODELS]
            
        plan.candidate_models = valid_candidates
        
        if not plan.candidate_models:
            raise ValueError("LLM failed to select any valid installed candidate models.")

    def _format_summary(self, plan: ExperimentPlan) -> str:
        
        cands = "\n".join([
            f"{i+1}. {c.model_name} (Priority: {c.priority})\n   Role/Rationale: {c.rationale}"
            for i, c in enumerate(plan.candidate_models)
        ])
        
        return f"""=========================================
MLFORGE MODEL ENGINEERING
=========================

Problem:
{plan.problem_description}

Task:
{plan.task_type}

Target:
{plan.target_column}

Dataset:
{plan.dataset_reference}

Primary Metric:
{plan.primary_metric}

=========================================
MODEL CANDIDATES
================

{cands}

=========================================
VALIDATION
==========

Strategy:
{plan.validation_strategy}

Primary metric:
{plan.primary_metric}

Secondary:
{', '.join(plan.secondary_metrics)}

=========================================

No models have been trained yet.

========================================="""
