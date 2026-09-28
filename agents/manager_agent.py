import json
from typing import Optional
from pydantic import BaseModel, Field

from core.llm_manager import LLMManager
from core.state.manager import AgentStateManager
from agents.handoff import Handoff
from agents.base_agent import BaseAgent
from core.orchestration.orchestrator import Orchestrator
from core.orchestration.models import MLProblem

class ManagerDecision(BaseModel):
    action: str = Field(description="Must be 'answer_directly', 'delegate', or 'orchestrate'")
    task: Optional[str] = Field(None, description="The task instructions for the worker (if delegate) or problem statement (if orchestrate)")
    answer: Optional[str] = Field(None, description="The final answer to the user, if answering directly")

class ManagerAgent(BaseAgent):
    def __init__(self, llm_provider: LLMManager, state_manager: AgentStateManager, worker_agent, orchestrator: Optional[Orchestrator] = None):
        super().__init__(
            llm_provider=llm_provider,
            name="manager",
            role="manager",
            system_prompt=(
                "You are a Manager Agent. You coordinate user requests. "
                "You can either answer the user directly if it's a simple conversation, delegate specific work (like calculations, reading files, or executing python) to a Worker agent, "
                "or orchestrate a full ML workflow. "
                "If delegating, set action='delegate' and provide a clear 'task' description for the worker. "
                "If orchestrating a multi-step ML workflow, set action='orchestrate' and provide the full ML 'task' problem description. "
                "If answering directly, set action='answer_directly' and provide the 'answer'."
            ),
            state=state_manager
        )
        self.state_manager = state_manager
        self.worker_agent = worker_agent
        self.orchestrator = orchestrator
        self.handoff = Handoff(self.state_manager)

    def run(self, user_input: str) -> str:
        self.state_manager.set_user_task(user_input)
        self.state_manager.add_message("user", user_input)
        print("[Manager] Request received")
        self.state_manager.add_decision(action="task_received", reason="User input received")

        context = self.state_manager.build_context()
        prompt = f"{context}\n\nUser Request: {user_input}\nDecide whether to 'delegate', 'orchestrate', or 'answer_directly'."
        
        try:
            resp = self.llm_provider.generate(prompt=prompt, system_prompt=self.system_prompt, response_schema=ManagerDecision)
            decision = ManagerDecision.model_validate_json(resp.content)
        except Exception as e:
            return f"Manager encountered an error: {e}"

        if decision.action == "answer_directly":
            print("[Manager] Generating final answer")
            self.state_manager.add_decision(action="final_answer_generated", reason="Answered without delegation")
            ans = decision.answer or "I have nothing to say."
            self.state_manager.add_message("assistant", ans)
            return ans
            
        elif decision.action == "orchestrate":
            print("[Manager] Orchestrating ML workflow")
            self.state_manager.add_decision(action="orchestration_required", reason="Task is a full ML workflow")
            
            problem = MLProblem(problem_statement=decision.task or user_input)
            workflow_result = self.orchestrator.run_workflow(problem)
            
            print("[Manager] Generating final answer")
            second_prompt = (
                f"User Request: {user_input}\n"
                f"Workflow Result: {workflow_result.model_dump_json()}\n\n"
                f"Generate the final 'answer' summarizing the workflow result. Return action='answer_directly'."
            )
            
            try:
                resp2 = self.llm_provider.generate(prompt=second_prompt, system_prompt=self.system_prompt, response_schema=ManagerDecision)
                dec2 = ManagerDecision.model_validate_json(resp2.content)
                self.state_manager.add_decision(action="final_answer_generated", reason="Answered based on workflow result")
                ans = dec2.answer or "I have nothing to say."
                self.state_manager.add_message("assistant", ans)
                return ans
            except Exception as e:
                return f"Manager error generating final answer: {e}"
                
        elif decision.action == "delegate":
            print("[Manager] Delegating task to Worker")
            self.state_manager.add_decision(action="delegation_required", reason="Task requires worker tools")
            self.state_manager.add_decision(action="worker_selected", reason="Selected the default worker")
            
            worker_result = self.handoff.delegate_to_worker(
                worker_agent=self.worker_agent,
                task_description=decision.task or user_input,
                context=context
            )
            
            self.state_manager.add_decision(action="worker_completed", reason="Worker returned result")
            
            print("[Manager] Generating final answer")
            second_prompt = (
                f"User Request: {user_input}\n"
                f"Worker Result: {worker_result.model_dump_json()}\n\n"
                f"Generate the final 'answer' for the user based on the worker's result. Return action='answer_directly'."
            )
            
            try:
                resp2 = self.llm_provider.generate(prompt=second_prompt, system_prompt=self.system_prompt, response_schema=ManagerDecision)
                dec2 = ManagerDecision.model_validate_json(resp2.content)
                self.state_manager.add_decision(action="final_answer_generated", reason="Answered based on worker result")
                ans = dec2.answer or "I have nothing to say."
                self.state_manager.add_message("assistant", ans)
                return ans
            except Exception as e:
                return f"Manager error generating final answer: {e}"
        else:
            return f"Unknown action: {decision.action}"
