from typing import Any, Dict, Optional, Type
from pydantic import BaseModel

class ToolResult(BaseModel):
    success: bool
    tool_name: str
    result: Any
    error: Optional[str] = None

class BaseTool:
    name: str = "base_tool"
    description: str = "Base tool description"
    args_schema: Type[BaseModel]

    def execute(self, **kwargs) -> Any:
        raise NotImplementedError("Tools must implement the execute method.")
