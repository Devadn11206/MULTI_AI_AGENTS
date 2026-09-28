import os
from typing import Optional
from core.state.models import AgentState, Message, Decision, Plan, DatasetInfo, Experiment, ModelInfo, EvaluationResult, ErrorInfo
from tools.base_tool import ToolResult

class AgentStateManager:
    def __init__(self):
        self.state: AgentState = AgentState()

    def create_state(self):
        self.state = AgentState()

    def get_state(self) -> AgentState:
        return self.state

    def set_user_task(self, task: str):
        self.state.user_task = task

    def add_message(self, role: str, content: str, metadata: Optional[dict] = None):
        msg = Message(role=role, content=content, metadata=metadata)
        self.state.messages.append(msg)

    def add_decision(self, action: str, tool_name: Optional[str] = None, reason: Optional[str] = None):
        dec = Decision(action=action, tool_name=tool_name, reason=reason)
        self.state.decisions.append(dec)

    def add_tool_result(self, result: ToolResult):
        self.state.tool_results.append(result)

    def set_plan(self, plan: Plan):
        self.state.current_plan = plan

    def set_dataset_info(self, dataset: DatasetInfo):
        self.state.dataset_info = dataset

    def add_experiment(self, experiment: Experiment):
        self.state.experiments.append(experiment)

    def set_best_model(self, model: ModelInfo):
        self.state.best_model = model

    def set_evaluation(self, eval_result: EvaluationResult):
        self.state.evaluation = eval_result

    def add_error(self, source: str, message: str, error_type: str):
        err = ErrorInfo(source=source, message=message, type=error_type)
        self.state.errors.append(err)

    def clear_state(self):
        self.create_state()

    def save_state(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.state.model_dump_json(indent=4))

    def load_state(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        self.state = AgentState.model_validate_json(data)

    def build_context(self) -> str:
        lines = []
        if self.state.user_task:
            lines.append(f"Current Task: {self.state.user_task}")
        
        if self.state.current_plan:
            lines.append(f"Goal: {self.state.current_plan.goal}")
        
        if self.state.tool_results:
            recent_results = self.state.tool_results[-3:]
            lines.append("Recent Tool Results:")
            for r in recent_results:
                status = "Success" if r.success else "Failed"
                out = r.result if r.success else r.error
                lines.append(f"- {r.tool_name} ({status}): {out}")
                
        lines.append("Conversation History:")
        for msg in self.state.messages[-10:]:
            lines.append(f"[{msg.role}] {msg.content}")
            
        return "\n".join(lines)
