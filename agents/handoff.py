from typing import Optional, List, Any
from pydantic import BaseModel, Field
import uuid
from core.state.manager import AgentStateManager

class WorkerTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: str = "manager"
    to_agent: str = "worker"
    task: str
    context: Optional[str] = None
    priority: str = "normal"

class WorkerResult(BaseModel):
    task_id: str
    agent: str = "worker"
    status: str
    result: Any
    errors: List[str] = Field(default_factory=list)

class Handoff:
    def __init__(self, state_manager: AgentStateManager):
        self.state_manager = state_manager

    def delegate_to_worker(self, worker_agent, task_description: str, context: Optional[str] = None) -> WorkerResult:
        task = WorkerTask(task=task_description, context=context)
        
        self.state_manager.add_message(
            role="manager",
            content=task_description,
            metadata={
                "sender": "manager",
                "recipient": "worker",
                "message_type": "task_assignment",
                "task_id": task.task_id
            }
        )
        print("[Handoff] Manager -> Worker")
        
        try:
            result = worker_agent.execute_task(task)
        except Exception as e:
            result = WorkerResult(
                task_id=task.task_id,
                status="failed",
                result="",
                errors=[str(e)]
            )
            
        
        content = result.result if result.status == "completed" else f"Failed: {result.errors}"
        if isinstance(content, dict) or isinstance(content, list):
            import json
            content = json.dumps(content)
        elif not isinstance(content, str):
            content = str(content)
            
        self.state_manager.add_message(
            role="worker",
            content=content,
            metadata={
                "sender": "worker",
                "recipient": "manager",
                "message_type": "task_result",
                "task_id": task.task_id
            }
        )
        print("[Handoff] Worker -> Manager")
        
        return result
