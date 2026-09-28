import sys
from core.llm_manager import LLMManager
from tools.registry import ToolRegistry
from tools.calculator import CalculatorTool
from tools.file_reader import FileReaderTool
from tools.python_executor import PythonExecutorTool
from tools.dataset_reader import DatasetReaderTool
from agents.manager_agent import ManagerAgent
from agents.worker_agent import WorkerAgent
from agents.data_analyst_agent import DataAnalystAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.model_engineering_agent import ModelEngineeringAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from core.state.manager import AgentStateManager
from core.orchestration.orchestrator import Orchestrator


def main():
    print("=========================================")
    print("MLFORGE")
    print("ORCHESTRATOR MODE")
    print("=========================================\n")

    try:
        from core.llm_manager import LLMManager
        provider = LLMManager()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Initialization Error: {e}")
        sys.exit(1)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FileReaderTool())
    registry.register(PythonExecutorTool())
    registry.register(DatasetReaderTool())

    state_manager = AgentStateManager()

    worker = WorkerAgent(llm_provider=provider, tool_registry=registry, state_manager=state_manager)
    data_analyst = DataAnalystAgent(llm_provider=provider, state_manager=state_manager)
    feature_engineer = FeatureEngineeringAgent(llm_provider=provider, state_manager=state_manager)
    model_engineer = ModelEngineeringAgent(llm_provider=provider, state_manager=state_manager)
    training_agent = TrainingAgent(llm_provider=provider, state_manager=state_manager)
    evaluation_agent = EvaluationAgent(llm_provider=provider, state_manager=state_manager)
    orchestrator = Orchestrator(
        llm_provider=provider,
        state_manager=state_manager,
        worker_agent=worker,
        data_analyst_agent=data_analyst,
        feature_engineering_agent=feature_engineer,
        model_engineering_agent=model_engineer,
        training_agent=training_agent,
        evaluation_agent=evaluation_agent,
    )
    manager = ManagerAgent(
        llm_provider=provider,
        state_manager=state_manager,
        worker_agent=worker,
        orchestrator=orchestrator,
    )

    print("Available tools (Worker Only):")
    for tool_name in registry.list_tools():
        print(f"* {tool_name}")

    while True:
        try:
            user_input = input("\nEnter your task:\n\n> ").strip().lstrip("> ")

            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye.")
                break

            if user_input.lower() in ["state", "/state"]:
                st = state_manager.get_state()
                print("\n[State Inspection]")
                print(f"Session: {st.session_id}")
                print(f"Task: {st.user_task}")
                print(f"Agents: manager, worker, data_analyst, orchestrator")
                print(f"Messages: {len(st.messages)}")
                print(f"Decisions: {len(st.decisions)}")
                if st.dataset_info:
                    di = st.dataset_info
                    print(f"\n[Dataset] {di.name} — {di.rows} rows x {di.columns} cols")
                    if di.target_column:
                        print(f"  Target: {di.target_column}")
                continue

            if user_input.lower() in ["/providers", "providers"]:
                print("\n=========================================")
                print("MLFORGE LLM PROVIDERS")
                print("=====================")
                print(f"\nPrimary:\n{provider.primary_name.capitalize()}")
                print(f"\nFallback:\n{', '.join([n.capitalize() for n in provider.fallback_names])}")
                print("\nConfigured:")
                for n in [provider.primary_name] + provider.fallback_names:
                    symbol = "[OK]" if n in provider.providers else "[FAIL]"
                    print(f"{symbol} {n.capitalize()}")
                continue

            if user_input.lower() in ["/test-provider", "/test-providers"]:
                print("\n")
                for n in [provider.primary_name] + provider.fallback_names:
                    print(f"Testing {n.capitalize()}...")
                    if n not in provider.providers:
                        print("[FAIL] Not configured\n")
                        continue
                    try:
                        resp = provider.providers[n].generate("Say hello", None, None)
                        print(f"[OK] {n.capitalize()} available\n")
                    except Exception as e:
                        print(f"[FAIL] Error: {e}\n")
                continue

            # /analyze-dataset <path> [target_column] or /analyze
            if user_input.lower().startswith("/analyze-dataset") or user_input.lower().startswith("/analyze"):
                parts = user_input.split()
                if len(parts) < 2:
                    print("Usage: /analyze-dataset <dataset_path> [target_column]")
                    continue
                dataset_path = parts[1]
                target_col = parts[2] if len(parts) > 2 else None
                print("\n=========================================")
                print("MLFORGE DATA ANALYST")
                print("====================\n")
                try:
                    profile = data_analyst.analyze(
                        dataset_path=dataset_path,
                        target_column=target_col,
                    )
                    print("\n" + data_analyst._format_summary(profile))
                except Exception as e:
                    print(f"\n[DataAgent] Error: {e}")
                continue

            # /engineer-features <path> <target_column>
            if user_input.lower().startswith("/engineer-features"):
                parts = user_input.split()
                if len(parts) < 3:
                    print("Usage: /engineer-features <dataset_path> <target_column>")
                    continue
                dataset_path = parts[1]
                target_col = parts[2]
                print("\n=========================================")
                print("MLFORGE FEATURE ENGINEERING")
                print("===========================\n")
                print(f"Dataset: {dataset_path}")
                print(f"Target:  {target_col}\n")
                # Try to use existing DatasetProfile from state if available
                existing_profile = None
                st = state_manager.get_state()
                if st.workflow_result and st.workflow_result.get("type") == "dataset_profile":
                    try:
                        from core.analysis.models import DatasetProfile
                        existing_profile = DatasetProfile(**st.workflow_result["data"])
                    except Exception:
                        pass
                try:
                    result = feature_engineer.engineer(
                        dataset_path=dataset_path,
                        target_column=target_col,
                        dataset_profile=existing_profile,
                    )
                    print("\n" + feature_engineer._format_summary(result))
                except Exception as e:
                    print(f"\n[FEAgent] Error: {e}")
                continue

            # /engineer-models <problem_description> <target_column>
            if user_input.lower().startswith("/engineer-models"):
                parts = user_input.split()
                if len(parts) < 3:
                    print("Usage: /engineer-models \"<problem_description>\" <target_column>")
                    continue
                # Simple split logic (handling quotes if needed)
                if '"' in user_input:
                    problem_desc = user_input.split('"')[1]
                    target_col = user_input.split('"')[2].strip()
                else:
                    problem_desc = parts[1]
                    target_col = parts[2]
                
                print("\n=========================================")
                print("MLFORGE MODEL ENGINEERING")
                print("===========================\n")
                print(f"Problem: {problem_desc}")
                print(f"Target:  {target_col}\n")
                
                try:
                    res = model_engineer.execute_task_from_step({
                        "step_id": "manual_cli",
                        "ml_problem": problem_desc,
                        "target_column": target_col
                    })
                    if res.status == "completed" and "summary" in res.result:
                        print("\n" + res.result["summary"])
                    else:
                        print(f"\n[MEAgent] Failed: {res.errors}")
                except Exception as e:
                    print(f"\n[MEAgent] Error: {e}")
                continue
                
            # /train or /run-experiments
            if user_input.lower() in ["/train", "/run-experiments"]:
                try:
                    res_train = training_agent.execute_task_from_step({"step_id": "manual_train"})
                    if res_train.status == "completed" and "summary" in res_train.result:
                        print("\n" + res_train.result["summary"])
                    else:
                        print(f"\n[TrainingAgent] Failed: {res_train.errors}")
                        continue
                        
                    res_eval = evaluation_agent.execute_task_from_step({"step_id": "manual_eval"})
                    if res_eval.status == "completed" and "summary" in res_eval.result:
                        print("\n" + res_eval.result["summary"])
                    else:
                        print(f"\n[EvaluationAgent] Failed: {res_eval.errors}")
                        
                except Exception as e:
                    print(f"\n[Training/Evaluation] Error: {e}")
                continue

            if user_input.lower() in ["/providers", "/test-provider", "/test-providers"]:
                continue

            if not user_input:
                continue

            print("")
            response = manager.run(user_input)
            print("\nAnswer:")
            print(response)

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except EOFError:
            print("\nGoodbye.")
            break


if __name__ == "__main__":
    main()
