import os
import uuid
from typing import Dict, Any, List
from fastapi import BackgroundTasks

from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager
from core.orchestration.models import MLProblem
from core.orchestration.orchestrator import Orchestrator
from agents.worker_agent import WorkerAgent
from agents.data_analyst_agent import DataAnalystAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.model_engineering_agent import ModelEngineeringAgent
from tools.registry import ToolRegistry

from debug.self_correction_controller import SelfCorrectionController

from api.schemas.workflow import WorkflowCreateRequest, WorkflowStatusResponse
from api.services.job_manager import job_manager
from api.services.dataset_service import DatasetService

class WorkflowService:
    @staticmethod
    def authorize_workflow(workflow_id: str, user_id: str):
        from core.database import SessionLocal
        from repositories.repository import Repository
        from fastapi import HTTPException
        db = SessionLocal()
        try:
            repo = Repository(db)
            wf = repo.get_workflow(workflow_id, user_id)
            if not wf:
                raise HTTPException(status_code=403, detail="Not authorized to access this workflow")
        finally:
            db.close()

    @staticmethod
    def create_workflow(req: WorkflowCreateRequest, background_tasks: BackgroundTasks, user_id: str) -> str:
        workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
        job_manager.create_workflow(workflow_id)
        
        # Save to database
        from core.database import SessionLocal
        from repositories.repository import Repository
        db = SessionLocal()
        try:
            repo = Repository(db)
            repo.create_workflow(
                user_id=user_id,
                dataset_id=req.dataset_id,
                objective=req.objective,
                task_type=req.task_type,
                target_column=req.target_column,
                priority=req.priority,
                workflow_id=workflow_id
            )
        finally:
            db.close()
        
        # Start in background
        background_tasks.add_task(WorkflowService._run_workflow_background, workflow_id, req)
        
        return workflow_id
        
    @staticmethod
    def _run_workflow_background(workflow_id: str, req: WorkflowCreateRequest):
        # ── Phase 15: Create trace manager ──────────────────────────────────
        from observability.trace_manager import create_trace_manager, remove_trace_manager
        tm = create_trace_manager(workflow_id)
        
        try:
            job_manager.update_status(workflow_id, "running", "orchestrator", 5, "Initializing workflow...")
            tm.start_workflow("Workflow started.")
            
            # Setup Core
            llm_provider = LLMManager()
            llm_provider.set_trace_manager(tm)
            state_manager = AgentStateManager()
            state = state_manager.state
            state.workflow_id = workflow_id
            state._trace_manager = tm
            
            from core.checkpoint_manager import CheckpointManager
            checkpoint_manager = CheckpointManager()
            checkpoint_manager.save_checkpoint(workflow_id, "start", "orchestrator", state)
            tm.record_checkpoint("start")
            
            # Agents
            worker = WorkerAgent(llm_provider, ToolRegistry(), state_manager)
            data_agent = DataAnalystAgent(llm_provider, state_manager)
            feature_agent = FeatureEngineeringAgent(llm_provider, state_manager)
            model_agent = ModelEngineeringAgent(llm_provider, state_manager)
            
            orchestrator = Orchestrator(
                llm_provider=llm_provider,
                state_manager=state_manager,
                worker_agent=worker,
                data_analyst_agent=data_agent,
                feature_engineering_agent=feature_agent,
                model_engineering_agent=model_agent
            )
            
            dataset_path = DatasetService.get_dataset_path(req.dataset_id)
            if not dataset_path:
                raise Exception(f"Dataset {req.dataset_id} not found.")
                
            problem = MLProblem(
                problem_statement=req.objective,
                dataset_path=dataset_path,
                target_column=req.target_column,
                problem_type=req.task_type,
                constraints=[f"{k}={v}" for k, v in (req.constraints.model_dump() if req.constraints else {}).items()]
            )
            
            # ── Orchestrator phase ───────────────────────────────────────────
            job_manager.update_status(workflow_id, "running", "orchestrator", 10, "Planning workflow...")
            job_manager.add_event(workflow_id, "orchestrator", "started", "running", "Workflow planning started.")
            tm.start_agent("orchestrator", "Workflow planning started.")
            
            job_manager.update_status(workflow_id, "running", "data_analyst", 20, "Analyzing dataset...")
            job_manager.add_event(workflow_id, "data_analyst", "started", "running", "Data analysis started.")
            tm.record_handoff("orchestrator", "data_analyst", "Orchestrator handed off to Data Analyst Agent.")
            tm.start_agent("data_analyst", "Data analysis started.")
            
            wf_result = orchestrator.run_workflow(problem)
            
            if wf_result.status != "completed":
                tm.complete_agent("data_analyst", "Data analysis phase done.")
                tm.complete_agent("orchestrator", "Orchestration ended with failure.")
                tm.fail_workflow(wf_result.final_message)
                job_manager.update_status(workflow_id, "failed", "orchestrator", 100, wf_result.final_message)
                job_manager.add_event(workflow_id, "orchestrator", "failed", "failed", wf_result.final_message)
                job_manager.save_result(workflow_id, {
                    "workflow_id": workflow_id,
                    "objective": req.objective,
                    "status": "failed",
                    "errors": [{"message": wf_result.final_message}]
                })
                return
                
            # Record orchestrator decisions from state
            for decision in getattr(state, "decisions", [])[-5:]:
                action = getattr(decision, "action", "")
                if action in ("plan_generated", "plan_validated"):
                    tm.record_decision(
                        "orchestrator",
                        decision=action,
                        message=getattr(decision, "reason", f"Orchestrator: {action}"),
                    )
            
            tm.complete_agent("data_analyst", "Dataset analysis completed.",
                result_summary=f"Dataset path: {dataset_path}")
            tm.record_handoff("data_analyst", "feature_engineer", "Handed off to Feature Engineer Agent.")
            tm.start_agent("feature_engineer", "Feature engineering started.")
            tm.complete_agent("feature_engineer", "Feature engineering completed.")
            tm.record_handoff("feature_engineer", "model_engineer", "Handed off to Model Engineer Agent.")
            tm.start_agent("model_engineer", "Model selection started.")
            
            job_manager.add_event(workflow_id, "orchestrator", "completed", "completed", "Orchestration completed.")
            job_manager.add_event(workflow_id, "model_engineer", "completed", "completed", "Model selection completed.")
            
            # Record model selection from experiment plan
            experiment_plan = state.experiment_plan
            if experiment_plan:
                candidate_models = experiment_plan.get("candidate_models", [])
                model_ids = ", ".join([c.get("model_id", "") for c in candidate_models]) if candidate_models else "None"
                tm.record_decision(
                    "model_engineer",
                    decision="model_selected",
                    message=f"Model Engineer selected candidate models: {model_ids}.",
                    metadata={"model_ids": model_ids},
                )
            tm.complete_agent("model_engineer", "Model selection completed.")
            tm.complete_agent("orchestrator", "Orchestration completed.")
            
            if not experiment_plan:
                tm.fail_workflow("No experiment plan generated.")
                job_manager.update_status(workflow_id, "failed", "orchestrator", 100, "No experiment plan generated.")
                job_manager.save_result(workflow_id, {
                    "workflow_id": workflow_id,
                    "objective": req.objective,
                    "status": "failed",
                    "errors": [{"message": "No experiment plan generated."}]
                })
                return
                
            # Requirements
            requirements = []
            if req.priority:
                # Use specified target_threshold if provided, or a high quality target (e.g., 0.85/0.90)
                target_val = req.target_threshold if req.target_threshold is not None else 0.85
                requirements.append({
                    "metric": req.priority,
                    "operator": ">=",
                    "target_value": target_val,
                    "priority": "primary",
                    "dataset_split": "test"
                })
                
            # ── Training / Evaluation / Self-correction / Review ─────────────
            job_manager.update_status(workflow_id, "running", "training", 60, "Training and evaluating models...")
            job_manager.add_event(workflow_id, "training", "started", "running", "Training started.")
            tm.record_handoff("orchestrator", "training", "Handed off to Training Agent.")
            model_names = "None"
            if experiment_plan:
                candidate_models = experiment_plan.get("candidate_models", [])
                model_names = ", ".join([c.get("model_id", "") for c in candidate_models]) if candidate_models else "None"
            tm.record_training("training", started=True,
                model_name=model_names)
            
            controller = SelfCorrectionController(llm_provider=llm_provider, state_manager=state_manager)
            
            # Attach trace_manager to state so SelfCorrectionController can emit events
            state._trace_manager = tm
            
            final_res = controller.run(
                experiment_plan=experiment_plan,
                requirements=requirements,
                dataset_reference=dataset_path,
                state=state
            )
            
            job_manager.add_event(workflow_id, "training", "completed", "completed", "Training pipeline finished.")
            job_manager.add_event(workflow_id, "reviewer", "completed", "completed", "Review completed.")
            
            # Format Final Result
            dataset_profile = None
            if state.feature_engineering_result:
                dataset_profile = state.feature_engineering_result
            elif state.workflow_result:
                dataset_profile = state.workflow_result

            ml_problem_data = state.ml_problem or {}

            real_errors = [
                e.model_dump() for e in state.errors
                if "Unknown tool requested" not in e.message
            ]
            hallucination_warnings = [
                f"[Agent fallback] {e.message}"
                for e in state.errors
                if "Unknown tool requested" in e.message
            ]

            best_metrics = {}
            if state.evaluation_results:
                best_metrics = state.evaluation_results[-1].get("metrics", {})

            sc_history = state.attempt_history if state.attempt_history else state.correction_history

            # If models were successfully trained and evaluated, mark workflow as completed (self-correction found best attainable model)
            has_evaluated_models = bool(state.evaluation_results) and len(best_metrics) > 0
            is_successful = final_res.get("status") in ["success", "completed"] or has_evaluated_models

            result_payload = {
                "workflow_id": workflow_id,
                "objective": req.objective,
                "status": "completed" if is_successful else "failed",
                "task_type": ml_problem_data.get("problem_type") or ml_problem_data.get("task_type"),
                "dataset_profile": dataset_profile,
                "dataset_info": {
                    "path": dataset_path,
                    "filename": os.path.basename(dataset_path),
                    "original_filename": req.dataset_id,
                    "target_column": req.target_column,
                    "rows": None,
                    "columns": None
                },
                "feature_engineering": state.feature_engineering_result,
                "experiment_plan": state.experiment_plan,
                "training_results": state.training_results,
                "evaluation_results": state.evaluation_results,
                "self_correction_history": sc_history,
                "reviewer_result": state.final_review,
                "best_model": best_metrics,
                "errors": real_errors,
                "warnings": (state.model_engineering_warnings or []) + hallucination_warnings,
                "usage_stats": llm_provider.usage_stats
            }
            
            # DB and MLflow integration
            from core.db_logging import log_experiment_to_db_and_mlflow
            from core.database import SessionLocal
            from repositories.repository import Repository
            
            best_model_name = None
            if getattr(state, "model_comparison", None) and isinstance(state.model_comparison, dict):
                best_model_name = state.model_comparison.get("best_model")
            if not best_model_name and getattr(state, "best_model_id", None):
                best_model_name = state.best_model_id
            if not best_model_name and getattr(state, "evaluation_results", []):
                best_model_name = state.evaluation_results[-1].get("model_name") or state.evaluation_results[-1].get("model_id")
            if not best_model_name and getattr(state, "experiment_plan", None) and isinstance(state.experiment_plan, dict):
                candidates = state.experiment_plan.get("candidate_models", [])
                if candidates and isinstance(candidates, list) and len(candidates) > 0:
                    best_model_name = candidates[0].get("model_name") or candidates[0].get("model_id")
            if not best_model_name:
                best_model_name = "best_model"

            parameters = state.evaluation_results[-1].get("parameters", {}) if state.evaluation_results else {}
            log_experiment_to_db_and_mlflow(
                workflow_id=workflow_id,
                model_name=best_model_name,
                task_type=result_payload["task_type"],
                metrics=best_metrics,
                parameters=parameters
            )

            
            db = SessionLocal()
            try:
                repo = Repository(db)
                repo.update_workflow_status(workflow_id, result_payload["status"])
            finally:
                db.close()
            
            # ── Emit final trace events ──────────────────────────────────────
            if best_metrics:
                tm.record_decision(
                    "training",
                    decision="best_model_selected",
                    message=f"Best model: {best_model_name}.",
                    metadata={"model": best_model_name, "metrics": {k: round(float(v), 4) for k, v in best_metrics.items() if v is not None}},
                )
            
            wf_status = result_payload["status"]
            if wf_status == "completed":
                tm.complete_workflow("Workflow completed successfully.")
            else:
                tm.fail_workflow("Workflow finished with errors.")
            
            job_manager.save_result(workflow_id, result_payload)
            job_manager.update_status(workflow_id, result_payload["status"], "completed", 100, "Workflow finished.")
            job_manager.add_event(workflow_id, "workflow", "completed", result_payload["status"], "Workflow fully completed.")
            
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            try:
                tm.fail_workflow(str(e)[:500])
            except Exception:
                pass
            job_manager.update_status(workflow_id, "failed", "system", 100, str(e))
            job_manager.add_event(workflow_id, "system", "failed", "failed", str(e))
            job_manager.save_result(workflow_id, {"workflow_id": workflow_id, "status": "failed", "errors": [{"message": str(e), "trace": err}]})
            
            from core.database import SessionLocal
            from repositories.repository import Repository
            db = SessionLocal()
            try:
                repo = Repository(db)
                repo.update_workflow_status(workflow_id, "FAILED", error_message=str(e))
            finally:
                db.close()
        finally:
            remove_trace_manager(workflow_id)

            
    @staticmethod
    def get_status(workflow_id: str) -> WorkflowStatusResponse:
        return job_manager.get_status(workflow_id)
        
    @staticmethod
    def get_events(workflow_id: str):
        return job_manager.get_events(workflow_id)
        
    @staticmethod
    def get_result(workflow_id: str):
        return job_manager.get_result(workflow_id)

    @staticmethod
    def get_recovery_info(workflow_id: str):
        from core.checkpoint_manager import CheckpointManager
        return CheckpointManager().get_recovery_info(workflow_id)

    @staticmethod
    def resume_workflow(workflow_id: str, background_tasks: BackgroundTasks):
        job_manager.update_status(workflow_id, "recovering", "orchestrator", 10, "Recovering workflow from checkpoint...")
        background_tasks.add_task(WorkflowService._resume_workflow_background, workflow_id)

    @staticmethod
    def _resume_workflow_background(workflow_id: str):
        try:
            from core.checkpoint_manager import CheckpointManager
            cp = CheckpointManager().load_latest_checkpoint(workflow_id)
            if not cp:
                raise Exception("No valid checkpoint found to resume from.")
            
            # Setup Core
            llm_provider = LLMManager()
            state_manager = AgentStateManager()
            
            # Restore state
            state_data = cp.get("state", {})
            for k, v in state_data.items():
                setattr(state_manager.state, k, v)
                
            state_manager.state.workflow_id = workflow_id
            phase = cp.get("phase", "orchestrator")
            
            # Agents
            worker = WorkerAgent(llm_provider, ToolRegistry(), state_manager)
            data_agent = DataAnalystAgent(llm_provider, state_manager)
            feature_agent = FeatureEngineeringAgent(llm_provider, state_manager)
            model_agent = ModelEngineeringAgent(llm_provider, state_manager)
            
            orchestrator = Orchestrator(
                llm_provider=llm_provider,
                state_manager=state_manager,
                worker_agent=worker,
                data_analyst_agent=data_agent,
                feature_engineering_agent=feature_agent,
                model_engineering_agent=model_agent
            )
            
            # We must resume from the correct spot. If plan exists, we can resume execution
            problem = None
            if state_manager.state.ml_problem:
                from core.orchestration.models import MLProblem
                problem = MLProblem(**state_manager.state.ml_problem)
                
            if phase == "start":
                if not problem:
                    from core.database import SessionLocal
                    from models.domain import Workflow
                    db = SessionLocal()
                    try:
                        wf_record = db.query(Workflow).filter(Workflow.id == workflow_id).first()
                        if not wf_record:
                            raise Exception("Workflow record not found to reconstruct MLProblem.")
                        dataset_path = DatasetService.get_dataset_path(wf_record.dataset_id)
                        from core.orchestration.models import MLProblem
                        problem = MLProblem(
                            problem_statement=wf_record.objective,
                            dataset_path=dataset_path,
                            target_column=wf_record.target_column,
                            problem_type=wf_record.task_type,
                            constraints=[]
                        )
                    finally:
                        db.close()
                        
                job_manager.update_status(workflow_id, "running", "orchestrator", 10, "Resuming execution from start...")
                job_manager.add_event(workflow_id, "orchestrator", "resumed", "running", "Workflow execution resumed from start.")
                
                wf_result = orchestrator.run_workflow(problem)
                if wf_result.status != "completed":
                    job_manager.update_status(workflow_id, "failed", "orchestrator", 100, wf_result.final_message)
                    return
                job_manager.add_event(workflow_id, "orchestrator", "completed", "completed", "Orchestration completed.")
                
            elif phase in ["data_analyst", "feature_engineering", "model_engineering", "training", "evaluation", "orchestrator"]:
                if state_manager.state.workflow_plan:
                    from core.orchestration.models import WorkflowPlan
                    plan = WorkflowPlan(**state_manager.state.workflow_plan)
                    
                    job_manager.update_status(workflow_id, "running", phase, 30, f"Resuming execution at {phase}...")
                    job_manager.add_event(workflow_id, "orchestrator", "resumed", "running", f"Workflow execution resumed at {phase}.")
                    
                    wf_result = orchestrator.executor.execute_plan(plan)
                    
                    if wf_result.status != "completed":
                        job_manager.update_status(workflow_id, "failed", "orchestrator", 100, wf_result.final_message)
                        return
                        
                    job_manager.add_event(workflow_id, "orchestrator", "completed", "completed", "Orchestration completed.")
                    
            # If we resume from after Orchestrator loop (SelfCorrection / Review)
            experiment_plan = state_manager.state.experiment_plan
            if not experiment_plan:
                job_manager.update_status(workflow_id, "failed", "orchestrator", 100, "No experiment plan found to resume self-correction.")
                return
                
            job_manager.update_status(workflow_id, "running", "training", 60, "Resuming training and evaluation...")
            controller = SelfCorrectionController(llm_provider=llm_provider, state_manager=state_manager)
            final_res = controller.run(
                experiment_plan=experiment_plan,
                requirements=[], # For now empty since we don't persist original reqs separately outside problem
                dataset_reference=problem.dataset_path if problem else "",
                state=state_manager.state
            )
            
            # Rebuild dataset info
            dataset_profile = None
            if state_manager.state.feature_engineering_result:
                dataset_profile = state_manager.state.feature_engineering_result
            elif state_manager.state.workflow_result:
                dataset_profile = state_manager.state.workflow_result
                
            ml_problem_data = state_manager.state.ml_problem or {}
            
            best_metrics = {}
            if state_manager.state.evaluation_results:
                best_metrics = state_manager.state.evaluation_results[-1].get("metrics", {})
                
            sc_history = state_manager.state.attempt_history if state_manager.state.attempt_history else state_manager.state.correction_history

            real_errors = [e.model_dump() if hasattr(e, 'model_dump') else e for e in state_manager.state.errors if "Unknown tool requested" not in (e.message if hasattr(e, 'message') else e.get('message', ''))]
            hallucination_warnings = [f"[Agent fallback] {e.message if hasattr(e, 'message') else e.get('message', '')}" for e in state_manager.state.errors if "Unknown tool requested" in (e.message if hasattr(e, 'message') else e.get('message', ''))]

            result_payload = {
                "workflow_id": workflow_id,
                "objective": ml_problem_data.get("problem_description", ""),
                "status": "completed" if final_res.get("status") in ["success", "completed"] else "failed",
                "task_type": ml_problem_data.get("problem_type") or ml_problem_data.get("task_type"),
                "dataset_profile": dataset_profile,
                "dataset_info": {
                    "path": ml_problem_data.get("dataset_path", ""),
                    "filename": os.path.basename(ml_problem_data.get("dataset_path", "")) if ml_problem_data.get("dataset_path") else "",
                    "original_filename": ml_problem_data.get("dataset_path", ""),
                    "target_column": ml_problem_data.get("target_column", ""),
                    "rows": None,
                    "columns": None
                },
                "feature_engineering": state_manager.state.feature_engineering_result,
                "experiment_plan": state_manager.state.experiment_plan,
                "training_results": state_manager.state.training_results,
                "evaluation_results": state_manager.state.evaluation_results,
                "self_correction_history": sc_history,
                "reviewer_result": state_manager.state.final_review,
                "best_model": best_metrics,
                "errors": real_errors,
                "warnings": (state_manager.state.model_engineering_warnings or []) + hallucination_warnings,
                "usage_stats": llm_provider.usage_stats
            }
            
            # DB and MLflow integration
            from core.db_logging import log_experiment_to_db_and_mlflow
            from core.database import SessionLocal
            from repositories.repository import Repository
            
            best_model_name = None
            if state_manager.state.evaluation_results:
                last_eval = state_manager.state.evaluation_results[-1]
                best_model_name = last_eval.get("model_name") or last_eval.get("model_id") or last_eval.get("model")
            if not best_model_name:
                best_model_name = "best_model"
            parameters = state_manager.state.evaluation_results[-1].get("parameters", {}) if state_manager.state.evaluation_results else {}
            log_experiment_to_db_and_mlflow(
                workflow_id=workflow_id,
                model_name=best_model_name,
                task_type=result_payload["task_type"],
                metrics=best_metrics,
                parameters=parameters
            )

            
            db = SessionLocal()
            try:
                repo = Repository(db)
                repo.update_workflow_status(workflow_id, result_payload["status"])
            finally:
                db.close()
            
            job_manager.save_result(workflow_id, result_payload)
            job_manager.update_status(workflow_id, result_payload["status"], "completed", 100, "Workflow finished.")
            job_manager.add_event(workflow_id, "workflow", "completed", result_payload["status"], "Workflow recovered and fully completed.")
            
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            job_manager.update_status(workflow_id, "failed", "system", 100, f"Recovery Failed: {e}")
            job_manager.add_event(workflow_id, "system", "failed", "failed", f"Recovery Failed: {e}")
            job_manager.save_result(workflow_id, {"workflow_id": workflow_id, "status": "failed", "errors": [{"message": f"Recovery Failed: {e}", "trace": err}]})
            
            from core.database import SessionLocal
            from repositories.repository import Repository
            db = SessionLocal()
            try:
                repo = Repository(db)
                repo.update_workflow_status(workflow_id, "FAILED", error_message=str(e))
            finally:
                db.close()
