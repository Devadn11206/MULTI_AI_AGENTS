from typing import Dict, List, Any
from tools.base_tool import BaseTool

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise ValueError(f"Unknown tool requested: {name}")
        return self._tools[name]

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = []
        for name, tool in self._tools.items():
            schemas.append({
                "name": name,
                "description": tool.description,
                "arguments": tool.args_schema.model_json_schema()
            })
        return schemas
