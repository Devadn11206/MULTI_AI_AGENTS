import sys
import io
import contextlib
import ast
from pydantic import BaseModel
from tools.base_tool import BaseTool

class PythonExecutorArgs(BaseModel):
    code: str

class PythonExecutorTool(BaseTool):
    name = "python_executor"
    description = "DEVELOPMENT ONLY: Executes safe basic Python code and returns output. Cannot use system commands or imports."
    args_schema = PythonExecutorArgs

    def execute(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise ValueError("Imports are blocked for security.")
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        allowed_funcs = {"print", "len", "range", "str", "int", "float", "list", "dict", "bool", "sum", "min", "max", "abs"}
                        if node.func.id not in allowed_funcs:
                            raise ValueError(f"Function call '{node.func.id}' is blocked.")
        except SyntaxError as e:
            raise ValueError(f"Syntax Error: {e}")
            
        safe_globals = {
            "__builtins__": {
                "print": print, "len": len, "range": range,
                "str": str, "int": int, "float": float,
                "list": list, "dict": dict, "bool": bool,
                "sum": sum, "min": min, "max": max, "abs": abs
            }
        }
        
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, safe_globals, {})
            result = stdout.getvalue()
            if not result:
                return "Code executed successfully with no output."
            return result
        except Exception as e:
            raise ValueError(f"Execution Error: {e}")
