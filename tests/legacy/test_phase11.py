import os
import json
import pandas as pd
from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager
from debug.self_correction_controller import SelfCorrectionController
from tools.review.review_engine import ReviewEngine

def create_dummy_dataset():
    data = {
        "feature1": [1.0, 2.0, 1.5, 8.0, 9.0, 8.5, 2.0, 1.1, 9.1, 8.8, 1.0, 1.2, 1.3, 1.4, 1.5],
        "feature2": [1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 1.0, 1.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "Target":   [0,   0,   0,   1,   1,   1,   0,   0,   1,   1,   0,   0,   0,   0,   0]
    }
    df = pd.DataFrame(data)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/dummy_review_test.csv", index=False)
    return "data/processed/dummy_review_test.csv"

def main():
    print("=========================================")
    print("TESTING MLFORGE FINAL REVIEWER (PHASE 11)")
    print("=========================================")
    
    llm_provider = LLMManager()
    state_manager = AgentStateManager()
    state = state_manager.state
    
    dataset_path = create_dummy_dataset()
    
    experiment_plan = {
        "experiment_id": "exp_phase11_test",
        "task_type": "classification",
        "target_column": "Target",
        "dataset_reference": dataset_path,
        "random_state": 42,
        "problem_description": "Test reviewer",
        "primary_metric": "recall",
        "validation_strategy": "train_test_split",
        "selection_reasoning": "Dummy",
        "baseline_model": "logistic_regression",
        "candidate_models": [
            {
                "model_id": "logistic_regression",
                "model_name": "Logistic Regression Baseline",
                "task_type": "classification",
                "priority": 1,
                "rationale": "Baseline"
            }
        ]
    }
    state.experiment_plan = experiment_plan
    
    # Let's set a requirement that is easy to pass so we get to Reviewer
    requirements = [
        {
            "metric": "recall",
            "operator": ">=",
            "target_value": 0.5,
            "priority": "primary",
            "dataset_split": "test"
        }
    ]
    
    controller = SelfCorrectionController(llm_provider=llm_provider, state_manager=state_manager)
    result = controller.run(
        experiment_plan=experiment_plan,
        requirements=requirements,
        dataset_reference=dataset_path,
        state=state
    )
    
    print("\n=========================================")
    print("FINAL RESULT INCORPORATING REVIEWER")
    print("=========================================")
    print(json.dumps(result, indent=2))
    
    if result.get("review_results"):
        print("\nREVIEW REPORT:")
        print("Status:", result["review_results"].get("status"))
        print("Score:", result["review_results"].get("overall_score"))

if __name__ == "__main__":
    main()
