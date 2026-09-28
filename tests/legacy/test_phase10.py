import os
import json
import pandas as pd
from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager
from debug.self_correction_controller import SelfCorrectionController

def create_dummy_dataset():
    # A heavily imbalanced AND overlapping dataset.
    # The features for class 0 and class 1 are almost identical, heavily biased toward class 0.
    # A standard Logistic Regression will predict 0 for everything, resulting in 0.0 Recall for class 1.
    # This will purposefully trigger a failure, forcing the Debugger Agent to intervene!
    data = {
        "feature1": [1.0, 1.2, 1.1, 1.3, 1.5, 1.0, 1.2, 1.1, 1.4, 1.5, 1.2, 1.3, 1.4, 1.1, 1.5],
        "feature2": [1.0, 1.0, 1.1, 1.2, 1.0, 1.1, 1.0, 1.2, 1.0, 1.1, 1.0, 1.0, 1.2, 1.1, 1.0],
        "Target":   [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1]
    }
    df = pd.DataFrame(data)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/dummy_test.csv", index=False)
    return "data/processed/dummy_test.csv"

def main():
    print("=========================================")
    print("TESTING MLFORGE SELF-CORRECTION (PHASE 10)")
    print("=========================================")
    
    # 1. Setup providers and state
    llm_provider = LLMManager()
    state_manager = AgentStateManager()
    state = state_manager.state
    
    dataset_path = create_dummy_dataset()
    
    # 2. Setup a dummy experiment plan
    experiment_plan = {
        "experiment_id": "exp_phase10_test",
        "task_type": "classification",
        "target_column": "Target",
        "dataset_reference": dataset_path,
        "random_state": 42,
        "problem_description": "Test self correction",
        "primary_metric": "recall",
        "validation_strategy": "train_test_split",
        "selection_reasoning": "Dummy selection",
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
    
    # 3. Setup Requirements 
    requirements = [
        {
            "metric": "recall",
            "operator": ">=",
            "target_value": 0.99,
            "priority": "primary",
            "dataset_split": "test"
        }
    ]
    
    # 4. Initialize Controller
    print(f"\nRequirement: Recall >= {requirements[0]['target_value']}")
    print(f"Initial Model: Logistic Regression")
    print(f"Dataset: {dataset_path}\n")
    
    controller = SelfCorrectionController(llm_provider=llm_provider, state_manager=state_manager)
    
    # 5. Run Self-Correction Loop
    result = controller.run(
        experiment_plan=experiment_plan,
        requirements=requirements,
        dataset_reference=dataset_path,
        state=state
    )
    
    # 6. Print Final Result
    print("\n=========================================")
    print("FINAL SELF-CORRECTION RESULT")
    print("=========================================")
    print(json.dumps(result, indent=2))
    
    print("\nDEBUG HISTORY:")
    for i, dbg in enumerate(state.debug_history):
        print(f"\nAttempt {i+1}:")
        print(json.dumps(dbg, indent=2))

if __name__ == "__main__":
    main()
