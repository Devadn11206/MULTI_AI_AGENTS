import json
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from core.llm_manager import LLMManager
from tools.registry import ToolRegistry
from tools.base_tool import ToolResult
from core.state.manager import AgentStateManager
from agents.handoff import WorkerTask, WorkerResult
from agents.base_agent import BaseAgent

class WorkerDecision(BaseModel):
    action: str = Field(description="Must be 'tool_call' or 'task_completed'")
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[str] = Field(None, description="The final result for the manager")

class WorkerAgent(BaseAgent):
    def __init__(self, llm_provider: LLMManager, tool_registry: ToolRegistry, state_manager: AgentStateManager):
        super().__init__(
            llm_provider=llm_provider,
            name="worker",
            role="worker",
            system_prompt=(
                "You are a specialized Worker Agent. Your job is to execute the specific task assigned by the Manager. "
                "You have access to tools. Always use the JSON schema provided. "
                "CRITICAL: When calling a tool, you MUST include ALL required arguments exactly as specified in the schema. "
                "For file_reader, you MUST include 'path' in arguments (e.g. {'path': 'datasets/sample.txt'}). "
                "If a tool is needed, return action='tool_call' with tool_name and arguments. "
                "If you can complete the task directly or after a tool call, return action='task_completed' and provide 'result'."
            ),
            state=state_manager
        )
        self.tool_registry = tool_registry
        self.state_manager = state_manager

    def run(self, user_input: str) -> str:
        raise NotImplementedError("WorkerAgent should be called via execute_task")

    def execute_task(self, task: WorkerTask) -> WorkerResult:
        print("[Worker] Task received")
        self.state_manager.add_decision(action="task_received", reason=f"Task {task.task_id} assigned to worker")
        
        schemas = self.tool_registry.get_tool_schemas()
        # Build explicit tool listing showing required arguments clearly
        tool_details = []
        for s in schemas:
            schema_props = s["arguments"].get("properties", {})
            required_args = s["arguments"].get("required", [])
            args_parts = []
            for k, v in schema_props.items():
                arg_type = v.get("type", "string")
                req = " (REQUIRED)" if k in required_args else " (optional)"
                args_parts.append(f'{k}: {arg_type}{req}')
            args_str = ", ".join(args_parts)
            tool_details.append(f'- {s["name"]}: {s["description"]}\n  Required arguments: {{ {args_str} }}')
        tools_str = "\n".join(tool_details)
        prompt = (
            f"Assigned Task: {task.task}\n"
            f"Context: {task.context}\n\n"
            f"Available Tools (CRITICAL: include ALL REQUIRED arguments):\n{tools_str}\n\n"
            f"Full schemas for reference:\n{json.dumps(schemas, indent=2)}"
        )
        
        try:
            resp = self.llm_provider.generate(prompt=prompt, system_prompt=self.system_prompt, response_schema=WorkerDecision)
            decision = WorkerDecision.model_validate_json(resp.content)
        except Exception as e:
            self.state_manager.add_decision(action="task_failed", reason="LLM failed to process task")
            return WorkerResult(task_id=task.task_id, status="failed", result="", errors=[str(e)])

        if decision.action == "task_completed":
            print("[Worker] Task completed")
            self.state_manager.add_decision(action="task_completed", reason="Worker finished without tools")
            return WorkerResult(task_id=task.task_id, status="completed", result=decision.result or "Done")

        elif decision.action == "tool_call":
            tool_name = decision.tool_name
            args = decision.arguments or {}
            print(f"[Worker] {tool_name.capitalize()} selected")
            print(f"[Tool] Executing {tool_name}...")
            
            try:
                tool = self.tool_registry.get_tool(tool_name)
                validated_args = tool.args_schema(**args)
                result_val = tool.execute(**validated_args.model_dump())
                tool_result = ToolResult(success=True, tool_name=tool_name, result=str(result_val))
            except Exception as e:
                tool_result = ToolResult(success=False, tool_name=tool_name, result=None, error=str(e))
                self.state_manager.add_error("Tool", str(e), "ToolExecutionError")
                
            self.state_manager.add_tool_result(tool_result)
            self.state_manager.add_decision(action="task_executed", reason=f"Executed {tool_name}")
            
            second_prompt = (
                f"Assigned Task: {task.task}\n"
                f"Tool executed: {tool_name}\n"
                f"Tool Output: {tool_result.result if tool_result.success else tool_result.error}\n\n"
                f"Return action='task_completed' with the final 'result' for the manager."
            )
            try:
                resp2 = self.llm_provider.generate(prompt=second_prompt, system_prompt=self.system_prompt, response_schema=WorkerDecision)
                dec2 = WorkerDecision.model_validate_json(resp2.content)
                print("[Worker] Task completed")
                self.state_manager.add_decision(action="task_completed", reason="Worker finished after tool execution")
                return WorkerResult(task_id=task.task_id, status="completed", result=dec2.result or "Done")
            except Exception as e:
                self.state_manager.add_decision(action="task_failed", reason="LLM failed on follow up")
                return WorkerResult(task_id=task.task_id, status="failed", result="", errors=[str(e)])
        
        else:
            err = f"Unknown action: {decision.action}"
            return WorkerResult(task_id=task.task_id, status="failed", result="", errors=[err])
