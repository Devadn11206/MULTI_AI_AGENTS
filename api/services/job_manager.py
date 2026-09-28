import uuid
import datetime
from typing import Dict, Any, List
from api.schemas.workflow import WorkflowStatusResponse, WorkflowEvent

class JobManager:
    def __init__(self):
        self.workflows: Dict[str, WorkflowStatusResponse] = {}
        self.events: Dict[str, List[WorkflowEvent]] = {}
        self.results: Dict[str, Any] = {}
        
    def create_workflow(self, workflow_id: str):
        self.workflows[workflow_id] = WorkflowStatusResponse(
            workflow_id=workflow_id,
            status="queued",
            current_agent="orchestrator",
            progress=0,
            message="Workflow queued."
        )
        self.events[workflow_id] = []
        
    def get_status(self, workflow_id: str) -> WorkflowStatusResponse:
        return self.workflows.get(workflow_id)
        
    def update_status(self, workflow_id: str, status: str, current_agent: str, progress: int, message: str):
        if workflow_id in self.workflows:
            wf = self.workflows[workflow_id]
            wf.status = status
            wf.current_agent = current_agent
            wf.progress = progress
            wf.message = message
            
    def add_event(self, workflow_id: str, agent: str, event_type: str, status: str, message: str, metadata: dict = None):
        if workflow_id not in self.events:
            self.events[workflow_id] = []
            
        event = WorkflowEvent(
            event_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            timestamp=datetime.datetime.utcnow().isoformat(),
            agent=agent,
            event_type=event_type,
            status=status,
            message=message,
            metadata=metadata
        )
        self.events[workflow_id].append(event)
        
    def get_events(self, workflow_id: str) -> List[WorkflowEvent]:
        return self.events.get(workflow_id, [])
        
    def save_result(self, workflow_id: str, result: Any):
        self.results[workflow_id] = result
        
    def get_result(self, workflow_id: str) -> Any:
        return self.results.get(workflow_id)

# Singleton instance
job_manager = JobManager()
