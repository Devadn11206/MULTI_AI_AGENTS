import os
import pandas as pd
from typing import Dict, Any

from agents.base_agent import BaseAgent
from agents.handoff import WorkerResult
from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager
from core.model_engineering.models import ExperimentPlan
from tools.ml.dataset_splitter import DatasetSplitter
from tools.ml.model_trainer import ModelTrainer

class TrainingAgent(BaseAgent):
    """Executes the ExperimentPlan by training models."""
    
    def __init__(self, llm_provider: LLMManager, state_manager: AgentStateManager):
        super().__init__(
            llm_provider=llm_provider,
            name="training_agent",
            role="training",
            system_prompt="You are the Training Agent. Summarize ML training execution.",
            state=state_manager
        )
        self.state_manager = state_manager
        self.trainer = ModelTrainer()

    def execute_task_from_step(self, step_info: Dict[str, Any]) -> WorkerResult:
        task_id = step_info.get("step_id", "training_step")
        
        st = self.state_manager.get_state()
        if not st.experiment_plan:
            return WorkerResult(task_id=task_id, agent=self.name, status="failed", result="", errors=["No experiment plan found in state."])
            
        plan = ExperimentPlan(**st.experiment_plan)
        
        try:
            # Load data
            if not os.path.exists(plan.dataset_reference):
                raise FileNotFoundError(f"Dataset not found: {plan.dataset_reference}")
            df = pd.read_csv(plan.dataset_reference)
            
            tm = getattr(st, "_trace_manager", None)
            tool_id = tm.start_tool("training_agent", "dataset_splitter") if tm else None
            try:
                # Split data
                X_train, X_test, y_train, y_test = DatasetSplitter.split(
                    df=df,
                    target_column=plan.target_column,
                    task_type=plan.task_type,
                    random_state=plan.random_state
                )
                if tm and tool_id:
                    tm.complete_tool("training_agent", "dataset_splitter", f"Split data: train={len(X_train)}, test={len(X_test)}", parent_event_id=tool_id)
            except Exception as split_err:
                if tm and tool_id:
                    tm.fail_tool("training_agent", "dataset_splitter", str(split_err), parent_event_id=tool_id)
                raise split_err
            
            # Save test set for EvaluationAgent
            test_path = f"artifacts/experiments/{plan.experiment_id}_test.csv"
            os.makedirs("artifacts/experiments", exist_ok=True)
            test_df = X_test.copy()
            target_name = plan.target_column if plan.target_column in df.columns else y_test.name
            test_df[target_name] = y_test
            test_df.to_csv(test_path, index=False)
            
            st.test_dataset_path = test_path
            
            # Train each model
            results = []
            for cand in plan.candidate_models:
                m_tool_id = tm.start_tool("training_agent", f"model_trainer_{cand.model_id}") if tm else None
                res = self.trainer.train(
                    experiment_id=plan.experiment_id,
                    model_id=cand.model_id,
                    model_name=cand.model_name,
                    task_type=plan.task_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    random_state=plan.random_state
                )
                if tm and m_tool_id:
                    if res.status == "completed":
                        tm.complete_tool("training_agent", f"model_trainer_{cand.model_id}", f"Trained {cand.model_name} in {res.training_time_seconds}s", parent_event_id=m_tool_id)
                    else:
                        tm.fail_tool("training_agent", f"model_trainer_{cand.model_id}", str(res.error), parent_event_id=m_tool_id)
                results.append(res.model_dump())
                
            # Store in state
            self.state_manager.state.training_results = results
            
            summary = self._format_summary(results, plan.experiment_id)
            return WorkerResult(
                task_id=task_id, 
                agent=self.name, 
                status="completed", 
                result={"type": "training_results", "summary": summary}
            )
            
        except Exception as e:
            return WorkerResult(task_id=task_id, agent=self.name, status="failed", result="", errors=[str(e)])
            
    def _format_summary(self, results, experiment_id: str) -> str:
        lines = [
            "=========================================", 
            "MLFORGE TRAINING", 
            "================\n",
            f"Experiment Plan: {experiment_id}",
            f"Models: {len(results)}\n",
            "Starting training...\n"
        ]
        
        for i, r in enumerate(results):
            lines.append(f"[{i+1}/{len(results)}] {r['model_name']}")
            if r['status'] == 'completed':
                lines.append(f"✓ Training complete ({r['training_time_seconds']}s)")
                lines.append(f"✓ Model saved")
            else:
                lines.append(f"✗ Failed: {r['error']}")
            lines.append("")
            
        return "\n".join(lines)
