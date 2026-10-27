import json
from typing import Dict, Any
from agents.base_agent import BaseAgent
from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager

class DebuggerAgent(BaseAgent):
    def __init__(self, llm_provider: LLMManager, state_manager: AgentStateManager):
        super().__init__(
            llm_provider=llm_provider,
            name="DebuggerAgent", 
            role="debugger",
            state=state_manager
        )
        self.state_manager = state_manager
        
    def execute(self, state: Any, *args, **kwargs) -> Any:
        attempts = getattr(state, "attempt_history", [])
        if not attempts:
            return {"status": "STOP", "message": "No attempts found to debug"}
            
        last_attempt = attempts[-1]
        
        system_prompt = """You are the Debugger Agent in MLForge.
Rules:
1. Analyze actual ML results.
2. Compare them with explicit user requirements.
3. Identify the most likely failure cause.
4. Recommend one controlled corrective action.
5. Only recommend actions from the Action Registry.
6. Never invent metrics.
7. Never invent models.
8. Never execute Python or shell commands.
9. Never modify project source code.
10. Never modify datasets directly.
11. Do not increase MAX_ITERATIONS.
12. Prefer the smallest corrective action first.
13. Return a structured DebugRecommendation.
14. If the requirement is already satisfied, return PASS.
15. If no safe correction exists, return STOP.

Schema for DebugRecommendation:
{
    "iteration": int,
    "status": "fail" | "pass" | "stop",
    "failure_type": str,
    "failed_requirement": str,
    "observed_result": float,
    "target_result": float,
    "diagnosis": str,
    "confidence": float,
    "recommended_action": str,
    "action_type": str,
    "affected_models": [str],
    "expected_effect": str,
    "safety_notes": str,
    "retry_allowed": bool,
    "rationale": str
}
"""
        
        context = {
            "requirements": getattr(state, "requirements", []),
            "last_attempt": last_attempt,
            "current_iteration": getattr(state, "current_iteration", 1),
            "max_iterations": getattr(state, "max_iterations", 3),
            "available_actions": [
                "class_weight", "threshold_optimization", "model_change", 
                "preprocessing_adjustment", "metric_strategy_change", "cross_validation_adjustment"
            ]
        }
        
        prompt = f"Analyze the following execution context and recommend a debug action in JSON format adhering to the DebugRecommendation schema.\n\nContext:\n{json.dumps(context, indent=2)}\n\n"
        
        response = self.llm_provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            response_schema=None,
            agent_name=self.name
        )
        
        try:
            # Attempt to parse json, possibly stripping backticks
            cleaned_response = response.content.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3]
            recommendation = json.loads(cleaned_response)
        except Exception as e:
            recommendation = {"status": "error", "message": f"Failed to parse debugger recommendation: {str(e)}", "action_type": None}
            
        return recommendation
