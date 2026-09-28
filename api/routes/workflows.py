from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from api.schemas.workflow import WorkflowCreateRequest, WorkflowCreateResponse, WorkflowStatusResponse, WorkflowEventsResponse
from api.services.workflow_service import WorkflowService
from api.dependencies.auth import get_current_user
from models.domain import User

router = APIRouter(prefix="/workflows", tags=["workflows"])

@router.post("", response_model=WorkflowCreateResponse, status_code=201)
def create_workflow(req: WorkflowCreateRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    workflow_id = WorkflowService.create_workflow(req, background_tasks, current_user.id)
    return {"workflow_id": workflow_id, "status": "queued"}

@router.get("/{workflow_id}", response_model=WorkflowStatusResponse)
def get_workflow_status(workflow_id: str, current_user: User = Depends(get_current_user)):
    # Authorize
    WorkflowService.authorize_workflow(workflow_id, current_user.id)
    
    status = WorkflowService.get_status(workflow_id)
    if not status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return status

@router.get("/{workflow_id}/events", response_model=WorkflowEventsResponse)
def get_workflow_events(workflow_id: str, current_user: User = Depends(get_current_user)):
    WorkflowService.authorize_workflow(workflow_id, current_user.id)
    events = WorkflowService.get_events(workflow_id)
    return {"events": events}

@router.post("/{workflow_id}/cancel")
def cancel_workflow(workflow_id: str, current_user: User = Depends(get_current_user)):
    WorkflowService.authorize_workflow(workflow_id, current_user.id)
    
    from api.services.job_manager import job_manager
    job_manager.update_status(workflow_id, "cancelled", "none", 100, "Workflow cancelled by user.")
    
    # Also update DB
    from core.database import SessionLocal
    from repositories.repository import Repository
    db = SessionLocal()
    try:
        repo = Repository(db)
        repo.update_workflow_status(workflow_id, "CANCELLED", error_message="Cancelled by user")
    finally:
        db.close()
        
    return {"status": "cancelled"}

@router.get("/recent/interrupted")
def get_recent_interrupted(current_user: User = Depends(get_current_user)):
    # Get DB workflows that belong to user and are RUNNING/QUEUED/DEGRADED but actually interrupted
    from core.database import SessionLocal
    from models.domain import Workflow
    from core.checkpoint_manager import CheckpointManager
    
    db = SessionLocal()
    cm = CheckpointManager()
    recoverable = []
    
    try:
        wfs = db.query(Workflow).filter(Workflow.user_id == current_user.id, Workflow.status.in_(["RUNNING", "QUEUED", "RECOVERING"])).all()
        for wf in wfs:
            info = cm.get_recovery_info(wf.id)
            if info:
                # Override phase with DB if desired, but we trust checkpoint
                recoverable.append(info)
    finally:
        db.close()
        
    recoverable.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return {"workflows": recoverable}

@router.get("/{workflow_id}/recovery")
def get_recovery_status(workflow_id: str, current_user: User = Depends(get_current_user)):
    WorkflowService.authorize_workflow(workflow_id, current_user.id)
    info = WorkflowService.get_recovery_info(workflow_id)
    if not info:
        raise HTTPException(status_code=404, detail="No recovery information found")
    return info

@router.post("/{workflow_id}/resume")
def resume_workflow(workflow_id: str, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    WorkflowService.authorize_workflow(workflow_id, current_user.id)
    info = WorkflowService.get_recovery_info(workflow_id)
    if not info:
        raise HTTPException(status_code=404, detail="No recovery information found")
    WorkflowService.resume_workflow(workflow_id, background_tasks)
    return {"workflow_id": workflow_id, "status": "recovering"}
