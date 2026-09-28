import pytest
import os
from tools.registry import ToolRegistry
from tools.calculator import CalculatorTool
from tools.file_reader import FileReaderTool
from tools.python_executor import PythonExecutorTool

def test_tool_registry():
    registry = ToolRegistry()
    calc = CalculatorTool()
    registry.register(calc)
    
    assert "calculator" in registry.list_tools()
    assert registry.get_tool("calculator") == calc
    
    with pytest.raises(ValueError, match="Unknown tool requested"):
        registry.get_tool("unknown")

def test_calculator_tool():
    calc = CalculatorTool()
    assert calc.execute("25 * 4 + 10") == "110"
    assert calc.execute("(10 + 5) * 3") == "45"
    assert calc.execute("2 ** 8") == "256"
    
    with pytest.raises(ValueError, match="is not allowed|Unsupported node type|Only numbers are allowed"):
        calc.execute("__import__('os').system('echo hello')")

def test_file_reader_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    reader = FileReaderTool()
    
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, World!")
    
    assert reader.execute("test.txt") == "Hello, World!"
    
    with pytest.raises(ValueError, match="Path traversal is not allowed"):
        reader.execute("../test.txt")
        
    with pytest.raises(ValueError, match="File not found"):
        reader.execute("missing.txt")
        
def test_python_executor_tool():
    executor = PythonExecutorTool()
    assert executor.execute("print('Hello World')").strip() == "Hello World"
    assert executor.execute("a = 5\nb = 10\nprint(a + b)").strip() == "15"
    
    with pytest.raises(ValueError, match="Imports are blocked"):
        executor.execute("import os\nos.system('echo hi')")
        
    with pytest.raises(ValueError, match="Function call 'exec' is blocked"):
        executor.execute("exec('print(1)')")
