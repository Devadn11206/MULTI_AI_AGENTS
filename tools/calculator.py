import ast
import operator
from pydantic import BaseModel
from tools.base_tool import BaseTool

class CalculatorArgs(BaseModel):
    expression: str

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Performs safe mathematical calculations. Input must be a valid mathematical expression using basic operators."
    args_schema = CalculatorArgs

    def execute(self, expression: str) -> str:
        allowed_operators = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.Mod: operator.mod,
            ast.USub: operator.neg, ast.UAdd: operator.pos
        }
        
        def _eval(node):
            if isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float)):
                    raise ValueError("Only numbers are allowed.")
                return node.value
            elif isinstance(node, ast.Num): # Python < 3.8 fallback
                return node.n
            elif isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                if type(node.op) not in allowed_operators:
                    raise ValueError(f"Operator {type(node.op).__name__} is not allowed.")
                if isinstance(node.op, ast.Pow) and right > 100:
                    raise ValueError("Exponent too large.")
                return allowed_operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                if type(node.op) not in allowed_operators:
                    raise ValueError(f"Operator {type(node.op).__name__} is not allowed.")
                return allowed_operators[type(node.op)](operand)
            elif isinstance(node, ast.Expression):
                return _eval(node.body)
            else:
                raise ValueError(f"Unsupported node type: {type(node).__name__}")
                
        try:
            tree = ast.parse(expression, mode='eval')
            result = _eval(tree.body)
            return str(result)
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")
