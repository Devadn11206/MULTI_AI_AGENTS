import os
from pydantic import BaseModel
from tools.base_tool import BaseTool
from config.settings import get_settings

class FileReaderArgs(BaseModel):
    path: str

class FileReaderTool(BaseTool):
    name = "file_reader"
    description = "Reads the contents of a text file. Path must be relative to the workspace directory."
    args_schema = FileReaderArgs

    def execute(self, path: str) -> str:
        settings = get_settings()
        workspace = os.path.abspath(getattr(settings, "WORKSPACE_DIR", os.getcwd()))
        
        if ".." in path:
            raise ValueError("Path traversal is not allowed.")
            
        full_path = os.path.abspath(os.path.join(workspace, path))
        
        if not full_path.startswith(workspace):
            raise ValueError("Access denied: Path is outside the approved workspace.")
            
        if not os.path.exists(full_path):
            raise ValueError(f"File not found: {path}")
            
        if not os.path.isfile(full_path):
            raise ValueError(f"Path is not a file: {path}")
            
        allowed_extensions = [".txt", ".md", ".csv", ".json", ".py", ".log"]
        if not any(full_path.endswith(ext) for ext in allowed_extensions):
            if "." in os.path.basename(full_path):
                raise ValueError("Unsupported file type.")
                
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read(100000) # Limit read size for safety
                return content
        except PermissionError:
            raise ValueError("Permission denied to read file.")
        except Exception as e:
            raise ValueError(f"Failed to read file: {e}")
