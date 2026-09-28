import asyncio
from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager
from agents.data_analyst_agent import DataAnalystAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.model_engineering_agent import ModelEngineeringAgent

def main():
    print("Testing MLForge Model Engineering End-to-End...")
    
    provider = LLMManager()
    state = AgentStateManager()
    
    analyst = DataAnalystAgent(provider, state)
    fe_agent = FeatureEngineeringAgent(provider, state)
    me_agent = ModelEngineeringAgent(provider, state)
    
    print("\n--- PHASE 6: Data Analysis ---")
    res1 = analyst.run("datasets/customer_churn.csv Churn")
    print(res1)
    
    print("\n--- PHASE 7: Feature Engineering ---")
    res2 = fe_agent.run("datasets/customer_churn.csv Churn")
    print(res2)
    
    print("\n--- PHASE 8: Model Engineering ---")
    res3 = me_agent.execute_task_from_step({
        "step_id": "test_e2e",
        "ml_problem": "Predict customer churn",
        "target_column": "Churn"
    })
    
    if res3.status == "completed":
        print("\n" + res3.result["summary"])
    else:
        print(f"\n[MEAgent] Failed: {res3.errors}")

if __name__ == "__main__":
    main()
