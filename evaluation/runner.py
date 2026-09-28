import os
import time
import argparse
import uuid
import json
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.database import SessionLocal, Base, engine
from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager
from core.orchestration.orchestrator import Orchestrator
from core.orchestration.models import MLProblem
from debug.self_correction_controller import SelfCorrectionController
from observability.trace_manager import TraceManager
from evaluation.schemas import EvaluationRunCreate, EvaluationResultCreate
from evaluation.repository import EvaluationRepository

from providers.base_provider import BaseProvider, LLMResponse

class MockProvider(BaseProvider):
    def __init__(self):
        super().__init__("mock_provider", "mock_model")
        
    def generate(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None, temperature: Optional[float] = None, response_schema: Optional[Any] = None, **kwargs) -> LLMResponse:
        # Very basic mock that mostly forces controlled failures
        return LLMResponse(
            content="{}",
            provider="mock_provider",
            model="mock_model",
            finish_reason="stop",
            usage={"input_tokens": 10, "output_tokens": 10}
        )

def run_workflow_test(
    test_id: str,
    dataset_path: str,
    target_column: str,
    task_type: str,
    objective: str,
    mode: str
) -> Dict[str, Any]:
    
    start_time = time.time()
    result = {
        "test_id": test_id,
        "dataset_id": os.path.basename(dataset_path),
        "status": "FAIL",
        "duration_ms": 0,
        "llm_calls": 0,
        "retries": 0,
        "fallbacks": 0,
        "best_model": None,
        "best_metric": None,
        "review_status": None,
        "error_code": None,
        "metadata_json": {}
    }
    
    # 1. Setup
    workflow_id = f"wf_eval_{uuid.uuid4().hex[:8]}"
    if mode == "mock":
        # Create a LLMManager with only our mock provider
        llm_provider = LLMManager()
        llm_provider.primary_provider = MockProvider()
        llm_provider.fallback_providers = []
    else:
        llm_provider = LLMManager()
        
    state_manager = AgentStateManager(workflow_id)
    tm = TraceManager(workflow_id)
    
    try:
        orchestrator = Orchestrator(llm_provider, state_manager)
        problem = MLProblem(
            problem_statement=objective,
            dataset_path=dataset_path,
            target_column=target_column,
            problem_type=task_type,
            constraints=[]
        )
        
        tm.start_agent("orchestrator", "Workflow planning started.")
        tm.start_agent("data_analyst", "Data analysis started.")
        
        # Run orchestrator
        wf_result = orchestrator.run_workflow(problem)
        
        state = state_manager.get_state()
        
        if wf_result.status != "completed":
            result["error_code"] = "ORCHESTRATOR_FAIL"
            result["metadata_json"]["final_message"] = wf_result.final_message
            raise Exception(f"Orchestrator failed: {wf_result.final_message}")
            
        experiment_plan = state.experiment_plan
        if not experiment_plan:
            result["error_code"] = "NO_PLAN"
            raise Exception("No experiment plan generated.")
            
        requirements = [{
            "metric": "accuracy" if task_type.endswith("classification") else "r2",
            "operator": ">=",
            "target_value": 0.5,
            "priority": "primary",
            "dataset_split": "test"
        }]
        
        state._trace_manager = tm
        controller = SelfCorrectionController(llm_provider, state_manager)
        
        # Run execution & evaluation
        final_res = controller.run(
            experiment_plan=experiment_plan,
            requirements=requirements,
            dataset_reference=dataset_path,
            state=state
        )
        
        if final_res.get("status") == "success":
            result["status"] = "PASS"
        else:
            result["status"] = "FAIL"
            result["error_code"] = "SELF_CORRECTION_FAIL"
            
        result["best_model"] = final_res.get("final_model_reference")
        
        metrics = final_res.get("final_metrics", {})
        if metrics:
            result["best_metric"] = metrics.get(requirements[0]["metric"])
            
        review_result = final_res.get("review_results", {})
        result["review_status"] = review_result.get("status") if isinstance(review_result, dict) else "unknown"
            
    except Exception as e:
        result["metadata_json"]["exception"] = str(e)
        result["metadata_json"]["traceback"] = traceback.format_exc()
        if not result["error_code"]:
            result["error_code"] = "EXCEPTION"
            
    # Gather stats
    try:
        history = llm_provider.get_request_history()
        result["llm_calls"] = len(history)
        result["retries"] = sum(1 for h in history if not h["success"])
        result["fallbacks"] = sum(1 for h in history if h.get("fallback_used", False))
    except:
        pass
        
    result["duration_ms"] = int((time.time() - start_time) * 1000)
    return result

def main():
    parser = argparse.ArgumentParser(description="MLForge Evaluation Runner")
    parser.add_argument("--mode", type=str, default="mock", choices=["mock", "live"])
    parser.add_argument("--tests", type=str, default="all")
    args = parser.parse_args()
    
    print(f"MLForge Evaluation Runner")
    print(f"Mode: {args.mode}")
    
    # Init DB
    Base.metadata.create_all(engine)
    db = SessionLocal()
    repo = EvaluationRepository(db)
    
    run_id = f"eval_{uuid.uuid4().hex[:8]}"
    run_record = repo.create_run(EvaluationRunCreate(
        evaluation_run_id=run_id,
        mode=args.mode,
        total_tests=0
    ))
    
    test_cases = [
        {"id": "tc_001_simple_cls", "ds": "data/eval/simple_classification.csv", "target": "target", "task": "binary_classification", "obj": "Predict class"},
        {"id": "tc_002_medium_reg", "ds": "data/eval/medium_regression.csv", "target": "target", "task": "regression", "obj": "Predict target"},
        {"id": "tc_003_messy_cls", "ds": "data/eval/messy_classification.csv", "target": "target", "task": "binary_classification", "obj": "Predict class robustly"},
        {"id": "tc_004_imb_cls", "ds": "data/eval/imbalanced_classification.csv", "target": "target", "task": "binary_classification", "obj": "Predict rare class"},
    ]
    
    # Filter datasets that actually exist
    valid_tests = []
    for t in test_cases:
        if os.path.exists(t["ds"]):
            valid_tests.append(t)
            
    print(f"Found {len(valid_tests)} valid test datasets.")
    
    passed = 0
    failed = 0
    start_time = time.time()
    
    for t in valid_tests:
        print(f"Running test {t['id']} on {t['ds']}...")
        res = run_workflow_test(
            test_id=t["id"],
            dataset_path=t["ds"],
            target_column=t["target"],
            task_type=t["task"],
            objective=t["obj"],
            mode=args.mode
        )
        
        repo.add_result(EvaluationResultCreate(
            evaluation_run_id=run_id,
            test_id=res["test_id"],
            test_category="workflow",
            dataset_id=res["dataset_id"],
            status=res["status"],
            duration_ms=res["duration_ms"],
            llm_calls=res["llm_calls"],
            retries=res["retries"],
            fallbacks=res["fallbacks"],
            best_model=res["best_model"],
            best_metric=res["best_metric"],
            review_status=res["review_status"],
            error_code=res["error_code"],
            metadata_json=res["metadata_json"]
        ))
        
        if res["status"] == "PASS":
            passed += 1
            print(f"  -> PASS ({res['duration_ms']}ms)")
        else:
            failed += 1
            print(f"  -> FAIL: {res['error_code']} ({res['duration_ms']}ms)")
            
    total_time = int((time.time() - start_time) * 1000)
    
    repo.update_run(run_id, {
        "completed_at": datetime.utcnow(),
        "total_tests": len(valid_tests),
        "passed_tests": passed,
        "failed_tests": failed,
        "duration_ms": total_time,
        "summary": {
            "workflow_success_rate": passed / max(1, len(valid_tests)),
            "message": "Evaluation completed."
        }
    })
    
    print("\n" + "="*50)
    print(f"MLForge Evaluation")
    print(f"Run ID: {run_id}")
    print(f"Workflow Reliability")
    print(f"Passed: {passed}/{len(valid_tests)}")
    print(f"Success Rate: {passed / max(1, len(valid_tests)) * 100:.1f}%")
    print(f"Average Execution Time: {total_time / max(1, len(valid_tests)) / 1000:.1f} sec")
    print("="*50)

if __name__ == "__main__":
    main()
