import json
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from core.llm_manager import LLMManager
from tools.registry import ToolRegistry
from tools.base_tool import ToolResult
from core.state.manager import AgentStateManager

class ToolCallDecision(BaseModel):
    action: str = Field(description="Must be 'tool_call' or 'final_answer'")
    tool_name: Optional[str] = Field(None, description="Name of the tool to use, if action is tool_call")
    arguments: Optional[Dict[str, Any]] = Field(None, description="Arguments for the tool, if action is tool_call")
    answer: Optional[str] = Field(None, description="The final answer to the user, if action is final_answer")

class ToolUsingAgent:
    def __init__(self, llm_provider: LLMManager, tool_registry: ToolRegistry, state_manager: AgentStateManager):
        self.llm_provider = llm_provider
        self.tool_registry = tool_registry
        self.state_manager = state_manager
        self.system_prompt = (
            "You are a helpful AI assistant with access to tools. "
            "Always respond using the structured schema provided. "
            "If you need to use a tool to fulfill the request, set action to 'tool_call', provide 'tool_name' and 'arguments'. "
            "If you can answer the user directly without tools, or you already have the tool result, set action to 'final_answer' and provide 'answer'."
        )

    def run(self, user_input: str) -> str:
        self.state_manager.set_user_task(user_input)
        self.state_manager.add_message("user", user_input)
        print("[State] User task recorded")
        
        schemas = self.tool_registry.get_tool_schemas()
        context = self.state_manager.build_context()
        
        prompt = f"{context}\n\nAvailable Tools:\n{json.dumps(schemas, indent=2)}"
        
        try:
            response_str = self.llm_provider.generate(
                prompt=prompt, 
                system_prompt=self.system_prompt,
                response_schema=ToolCallDecision
            )
            decision = ToolCallDecision.model_validate_json(response_str.content)
        except Exception as e:
            self.state_manager.add_error("Agent", str(e), "DecisionError")
            return f"Agent encountered an error deciding action: {e}"

        self.state_manager.add_decision(
            action=decision.action, 
            tool_name=decision.tool_name, 
            reason="LLM decided action based on context"
        )
        print("[State] Decision recorded")

        if decision.action == "final_answer":
            answer = decision.answer or "I have nothing to say."
            self.state_manager.add_message("assistant", answer)
            print("[State] Assistant response recorded")
            return answer

        elif decision.action == "tool_call":
            tool_name = decision.tool_name
            args = decision.arguments or {}
            print(f"[Agent] Selected tool: {tool_name}")
            
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
            self.state_manager.add_message("tool", tool_result.result if tool_result.success else tool_result.error, metadata={"tool_name": tool_name})
            print("[State] Tool result recorded")
            
            print("[Agent] Generating final answer...")
            second_context = self.state_manager.build_context()
            second_prompt = (
                f"{second_context}\n\n"
                f"Now provide the final answer using the structured schema."
            )
            
            try:
                second_response_str = self.llm_provider.generate(
                    prompt=second_prompt, 
                    system_prompt=self.system_prompt,
                    response_schema=ToolCallDecision
                )
                final_decision = ToolCallDecision.model_validate_json(second_response_str.content)
                answer = final_decision.answer or "I could not generate a final answer."
                self.state_manager.add_message("assistant", answer)
                print("[State] Assistant response recorded")
                return answer
            except Exception as e:
                self.state_manager.add_error("Agent", str(e), "FinalAnswerError")
                return f"Agent encountered an error generating final answer: {e}"
        else:
            err = f"Agent produced unknown action: {decision.action}"
            self.state_manager.add_error("Agent", err, "InvalidActionError")
            return err
