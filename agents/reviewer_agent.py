import json
from typing import Dict, Any
from agents.base_agent import BaseAgent
from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager
from tools.review.review_engine import ReviewEngine
from tools.review.review_reporter import ReviewReporter
from tools.review.models import ReviewResult

class ReviewerAgent(BaseAgent):
    def __init__(self, llm_provider: LLMManager, state_manager: AgentStateManager):
        super().__init__(
            llm_provider=llm_provider,
            name="ReviewerAgent", 
            role="reviewer",
            state=state_manager
        )
        self.state_manager = state_manager
        self.engine = ReviewEngine()
        
    def execute_task_from_step(self, step_info: Dict[str, Any]) -> Any:
        state = self.state_manager.get_state()
        
        # 1. Run deterministic review checks
        tm = getattr(state, "_trace_manager", None)
        rev_tool_id = tm.start_tool("reviewer", "review_engine") if tm else None
        review_result: ReviewResult = self.engine.run_review(state)
        if tm and rev_tool_id:
            tm.complete_tool("reviewer", "review_engine", f"Review status: {review_result.status}", parent_event_id=rev_tool_id)
        
        # 2. Get LLM summary and reasoning
        summary = self._generate_summary(review_result)
        review_result.reviewer_summary = summary
        
        # 3. Format Report
        report = ReviewReporter.generate_report(review_result)
        
        # 4. Save to state
        if not hasattr(state, "review_history"):
            state.review_history = []
        state.review_history.append(review_result.to_dict())
        state.final_review = review_result.to_dict()
        
        # Save state to disk for review
        self.state_manager.save_state("artifacts/experiments/final_review_state.json")
        
        return {
            "status": "completed" if review_result.status in ["PASS", "PASS_WITH_WARNINGS"] else "failed",
            "quality_gate_status": review_result.status,
            "report": report,
            "result_data": review_result.to_dict()
        }

    def _generate_summary(self, result: ReviewResult) -> str:
        system_prompt = (
            "You are the independent Reviewer Agent in MLForge.\n"
            "Your role is to critically review the final ML solution.\n"
            "You are independent from the agents that created the solution.\n"
            "Do not assume previous agents are correct.\n"
            "Use evidence from deterministic review tools.\n"
            "Never fabricate metrics.\n"
            "Never override a deterministic critical failure.\n"
        )
        
        prompt = f"Review the following deterministic checks and summarize the findings:\n{json.dumps(result.to_dict(), indent=2)}\n\nProvide a human-readable summary paragraph."
        
        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                response_schema=None,
                agent_name=self.name
            )
            return response.content.strip()
        except Exception as e:
            return f"LLM Summary unavailable. Deterministic checks concluded with status {result.status}."
