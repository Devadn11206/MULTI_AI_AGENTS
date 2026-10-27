import pandas as pd
from typing import Dict, Any

from agents.base_agent import BaseAgent
from agents.handoff import WorkerResult
from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager
from core.model_engineering.models import ExperimentPlan
from core.ml.models import TrainingResult
from tools.ml.model_evaluator import ModelEvaluator
from tools.ml.model_comparator import ModelComparator

class EvaluationAgent(BaseAgent):
    """Evaluates trained models and selects the best one."""
    
    def __init__(self, llm_provider: LLMManager, state_manager: AgentStateManager):
        super().__init__(
            llm_provider=llm_provider,
            name="evaluation_agent",
            role="evaluation",
            system_prompt="You are the Evaluation Agent. Summarize ML evaluation.",
            state=state_manager
        )
        self.state_manager = state_manager
        self.evaluator = ModelEvaluator()

    def execute_task_from_step(self, step_info: Dict[str, Any]) -> WorkerResult:
        task_id = step_info.get("step_id", "evaluation_step")
        st = self.state_manager.get_state()
        
        if not st.experiment_plan or not st.training_results or not hasattr(st, "test_dataset_path"):
            return WorkerResult(task_id=task_id, agent=self.name, status="failed", result="", errors=["Missing plan, training results, or test data."])
            
        plan = ExperimentPlan(**st.experiment_plan)
        training_results = [TrainingResult(**r) for r in st.training_results]
        
        try:
            # Load test data
            df = pd.read_csv(st.test_dataset_path)
            X_test = df.drop(columns=[plan.target_column])
            y_test = df[plan.target_column]
            
            tm = getattr(st, "_trace_manager", None)
            eval_results = []
            
            # Evaluate each model
            for tr in training_results:
                if tr.status != "completed":
                    continue
                e_tool_id = tm.start_tool("evaluation_agent", f"model_evaluator_{tr.model_id}") if tm else None
                try:
                    res = self.evaluator.evaluate(
                        training_result=tr,
                        primary_metric=plan.primary_metric,
                        X_test=X_test,
                        y_test=y_test
                    )
                    eval_results.append(res)
                    if tm and e_tool_id:
                        tm.complete_tool("evaluation_agent", f"model_evaluator_{tr.model_id}", f"Evaluated {tr.model_name}", parent_event_id=e_tool_id)
                except Exception as eval_err:
                    if tm and e_tool_id:
                        tm.fail_tool("evaluation_agent", f"model_evaluator_{tr.model_id}", str(eval_err), parent_event_id=e_tool_id)
                    raise eval_err
                
            self.state_manager.state.evaluation_results = [r.model_dump() for r in eval_results]
            
            # Compare models
            c_tool_id = tm.start_tool("evaluation_agent", "model_comparator") if tm else None
            comp = ModelComparator.compare(eval_results, plan.task_type, plan.primary_metric)
            if tm and c_tool_id:
                tm.complete_tool("evaluation_agent", "model_comparator", f"Best model: {comp.best_model}", parent_event_id=c_tool_id)
            self.state_manager.state.model_comparison = comp.model_dump()
            self.state_manager.state.best_model_id = comp.best_model
            
            summary = self._format_summary(plan, eval_results, comp)
            return WorkerResult(
                task_id=task_id, 
                agent=self.name, 
                status="completed", 
                result={"type": "evaluation_results", "summary": summary}
            )
            
        except Exception as e:
            return WorkerResult(task_id=task_id, agent=self.name, status="failed", result="", errors=[str(e)])
            
    def _format_summary(self, plan: ExperimentPlan, eval_results, comp) -> str:
        lines = [
            "=========================================",
            "MLFORGE MODEL EVALUATION",
            "========================\n",
            f"Problem:\n{plan.problem_description}\n",
            f"Task:\n{plan.task_type}\n",
            f"Primary Metric:\n{plan.primary_metric}\n",
            "=========================================",
            "MODEL RESULTS",
            "=============\n"
        ]
        
        # Get all keys for header
        all_metrics = set()
        for r in eval_results:
            if r.status == "completed":
                all_metrics.update(r.metrics.keys())
                
        sorted_metrics = sorted(list(all_metrics))
        header = "Model".ljust(25) + " ".join([m.rjust(10) for m in sorted_metrics])
        lines.append(header)
        
        for r in eval_results:
            if r.status != "completed":
                lines.append(f"{r.model_name.ljust(25)} FAILED: {r.error}")
                continue
                
            row = r.model_name.ljust(25)
            for m in sorted_metrics:
                val = r.metrics.get(m)
                row += f"{val:10.4f}" if val is not None else "       N/A"
            lines.append(row)
            
        lines.extend([
            "\n=========================================",
            "BEST MODEL",
            "==========\n",
            f"{comp.best_model or 'None'}\n",
            f"Primary Metric ({plan.primary_metric}):",
        ])
        
        if comp.ranking:
            lines.append(f"{comp.ranking[0]['primary_metric_value']:.4f}\n")
        else:
            lines.append("N/A\n")
            
        lines.append("=========================================\n")
        failed_eval = sum(1 for r in eval_results if r.status != "completed")
        lines.append(f"Evaluation failures: {failed_eval}")
        lines.append("=========================================")
        
        return "\n".join(lines)
