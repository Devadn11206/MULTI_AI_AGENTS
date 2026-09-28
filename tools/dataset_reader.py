import os
import pandas as pd
from pydantic import BaseModel
from tools.base_tool import BaseTool
from config.settings import get_settings


class DatasetReaderArgs(BaseModel):
    path: str


class DatasetReaderTool(BaseTool):
    name = "dataset_reader"
    description = (
        "Loads a tabular dataset (CSV) from a path relative to the workspace. "
        "Returns the DataFrame as a safe Python object for analysis. "
        "Required argument: path (str) — path relative to workspace directory."
    )
    args_schema = DatasetReaderArgs

    SUPPORTED_EXTENSIONS = [".csv"]

    def execute(self, path: str) -> pd.DataFrame:
        settings = get_settings()
        workspace = os.path.abspath(getattr(settings, "WORKSPACE_DIR", os.getcwd()))

        if ".." in path and not os.path.isabs(path):
            raise ValueError("Path traversal is not allowed.")

        if os.path.isabs(path) and os.path.exists(path):
            full_path = path
        else:
            full_path = os.path.abspath(os.path.join(workspace, path))
            if not os.path.exists(full_path):
                cwd_path = os.path.abspath(os.path.join(os.getcwd(), path))
                if os.path.exists(cwd_path):
                    full_path = cwd_path

        if not os.path.exists(full_path):
            raise ValueError(f"File not found: {path}")

        if not os.path.isfile(full_path):
            raise ValueError(f"Not a file: {path}")

        ext = os.path.splitext(full_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        try:
            if ext == ".csv":
                df = pd.read_csv(full_path, encoding="utf-8")
            else:
                raise ValueError(f"Unsupported extension: {ext}")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(full_path, encoding="latin-1")
            except Exception as e:
                raise ValueError(f"Encoding error reading file: {e}")
        except pd.errors.EmptyDataError:
            raise ValueError("The dataset file is empty.")
        except pd.errors.ParserError as e:
            raise ValueError(f"Malformed CSV file: {e}")
        except PermissionError:
            raise ValueError("Permission denied reading file.")
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")

        return df
