import uuid
import os
import json
from typing import Dict, Any, List
from debug.requirement_evaluator import RequirementEvaluator, MetricRequirement
from debug.action_registry import ActionRegistry
from agents.debugger_agent import DebuggerAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.reviewer_agent import ReviewerAgent

from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager

class SelfCorrectionController:
    def __init__(self, llm_provider: LLMManager, state_manager: AgentStateManager):
        self.llm_provider = llm_provider
        self.state_manager = state_manager
        self.action_registry = ActionRegistry()
        self.debugger = DebuggerAgent(llm_provider=self.llm_provider, state_manager=self.state_manager)
        self.reviewer = ReviewerAgent(llm_provider=self.llm_provider, state_manager=self.state_manager)
        
    def run(self, experiment_plan: Dict[str, Any], requirements: List[Dict[str, Any]], dataset_reference: str, state: Any) -> Dict[str, Any]:
        state.requirements = requirements
        state.current_iteration = 1
        
        # Phase 15: retrieve TraceManager if available (attached by WorkflowService)
        tm = getattr(state, '_trace_manager', None)
        
        while state.current_iteration <= state.max_iterations:
            iteration = state.current_iteration
            print(f"--- ATTEMPT {iteration} ---")
            
            # Step 1: Train
            candidate_models = experiment_plan.get("candidate_models", [])
            model_names = ", ".join([c.get("model_id", "") for c in candidate_models]) if candidate_models else "None"
            if tm:
                tm.record_training("training", started=True,
                    model_name=model_names,
                    iteration=iteration)
            training_agent = TrainingAgent(llm_provider=self.llm_provider, state_manager=self.state_manager)
            train_res = training_agent.execute_task_from_step({"step_id": f"train_{iteration}"})
            if tm:
                tm.record_training("training", started=False,
                    model_name=model_names,
                    iteration=iteration)
            
            # Step 2: Evaluate
            if tm:
                tm.record_evaluation("evaluation", started=True, iteration=iteration)
            eval_agent = EvaluationAgent(llm_provider=self.llm_provider, state_manager=self.state_manager)
            eval_res = eval_agent.execute_task_from_step({"step_id": f"eval_{iteration}"})
            if tm:
                tm.record_evaluation("evaluation", started=False, iteration=iteration)
            
            # Step 3: Check requirements
            req_results = []
            all_passed = True
            
            # Check for training/evaluation errors
            if getattr(state, "errors", []) and state.errors[-1].source in ["training", "evaluation"]:
                all_passed = False
                req_results.append({
                    "status": "fail", 
                    "message": f"{state.errors[-1].source.capitalize()} failed: {state.errors[-1].message}"
                })
                if tm:
                    tm.record_requirement(False, req_results[-1]["message"], iteration=iteration)
            else:
                metrics = state.evaluation_results[-1].get("metrics", {}) if getattr(state, "evaluation_results", []) else {}
                for req in requirements:
                    mr = MetricRequirement(**req)
                    res = RequirementEvaluator.evaluate(metrics, mr)
                    req_results.append(res.to_dict())
                    if res.status == "fail":
                        all_passed = False
                        if tm:
                            tm.record_requirement(False, res.to_dict().get("message", "Requirement failed."),
                                metadata={"metric": req.get("metric"), "actual": metrics.get(req.get("metric"))},
                                iteration=iteration)
                    else:
                        if tm:
                            tm.record_requirement(True, res.to_dict().get("message", "Requirement passed."),
                                metadata={"metric": req.get("metric"), "actual": metrics.get(req.get("metric"))},
                                iteration=iteration)
            
            state.requirement_results = req_results
            
            # Record Attempt
            attempt_info = {
                "attempt_number": iteration,
                "experiment_plan": experiment_plan.copy(),
                "training_results": state.training_results[-1] if getattr(state, "training_results", []) else {},
                "evaluation_results": state.evaluation_results[-1] if getattr(state, "evaluation_results", []) else {},
                "requirement_result": req_results,
                "status": "pass" if all_passed else "fail"
            }
            
            # Phase 11: Reviewer Gate
            if all_passed:
                print(f"--- RUNNING REVIEWER GATE ---")
                if tm:
                    tm.record_review(started=True)
                review_res = self.reviewer.execute_task_from_step({"step_id": f"review_{iteration}"})
                attempt_info["review_result"] = review_res
                
                reviewer_passed = (review_res["status"] != "failed")
                if tm:
                    summary = review_res.get("reviewer_summary") or review_res.get("summary", "")
                    checks = review_res.get("checks", [])
                    safe_findings = [c.get("name") for c in checks if isinstance(c, dict)][:5]
                    tm.record_review(
                        started=False,
                        passed=reviewer_passed,
                        message=(f"Reviewer approved the solution. Checks: {safe_findings}." if reviewer_passed
                                 else f"Reviewer rejected solution. Issues: {safe_findings}."),
                        metadata={"checks_count": len(checks), "findings": safe_findings},
                    )
                
                if review_res["status"] == "failed":
                    print("Reviewer rejected the solution. Sending to Debugger.")
                    all_passed = False
                    attempt_info["status"] = "fail"
                    
            state.attempt_history.append(attempt_info)
            
            if all_passed:
                state.self_correction_status = "success"
                state.best_attempt = iteration
                state.final_model_reference = state.best_model_id if hasattr(state, "best_model_id") else None
                break
                
            if state.current_iteration == state.max_iterations:
                state.self_correction_status = "failed"
                break
                
            # Step 4: Debugger
            debug_rec = self.debugger.execute(state)
            state.current_debug_recommendation = debug_rec
            state.debug_history.append(debug_rec)
            
            if tm:
                action_summary = debug_rec.get("action_type") or debug_rec.get("recommended_action", "")
                tm.record_debugger(
                    message=f"Debugger recommended: {action_summary}.",
                    decision=action_summary,
                    metadata={"action_type": action_summary,
                               "affected_models": debug_rec.get("affected_models", [])},
                    iteration=iteration,
                )
            
            if debug_rec.get("status", "").upper() == "STOP" or debug_rec.get("status", "").upper() == "PASS":
                state.self_correction_status = "failed" if debug_rec.get("status", "").upper() == "STOP" else "success"
                break
                
            action_type = debug_rec.get("action_type")
            if not action_type or not self.action_registry.get_action(action_type):
                print("Invalid action recommended. Stopping.")
                state.self_correction_status = "failed"
                break
                
            # Loop detection
            correction_sig = f"{action_type}_{experiment_plan.get('model_id')}"
            if correction_sig in [c.get("signature") for c in state.correction_history]:
                print("Loop detected. Stopping.")
                state.self_correction_status = "failed"
                break
                
            state.correction_history.append({
                "attempt": state.current_iteration,
                "action": action_type,
                "signature": correction_sig
            })
            
            # Apply correction
            if action_type == "class_weight":
                experiment_plan["parameters"] = experiment_plan.get("parameters", {})
                experiment_plan["parameters"]["class_weight"] = "balanced"
                state.experiment_plan = experiment_plan
            elif action_type == "threshold_optimization":
                # Concept implementation
                pass
            elif action_type == "model_change":
                new_model = debug_rec.get("recommended_action") # or another field
                # Example: Extract the new model name if provided in some format, or from 'affected_models'
                if debug_rec.get("affected_models") and len(debug_rec["affected_models"]) > 0:
                    new_model = debug_rec["affected_models"][0]
                    experiment_plan["model_id"] = new_model
                    state.experiment_plan = experiment_plan
            
            # Proceed to next iteration
            state.current_iteration += 1
            
        # Return final result
        return {
            "status": state.self_correction_status,
            "total_iterations": state.current_iteration,
            "best_attempt": getattr(state, "best_attempt", None),
            "final_metrics": state.evaluation_results[-1].get("metrics", {}) if getattr(state, "evaluation_results", []) else {},
            "requirements": requirements,
            "requirement_results": getattr(state, "requirement_results", []),
            "review_results": getattr(state, "final_review", None),
            "final_model_reference": getattr(state, "final_model_reference", None)
        }
