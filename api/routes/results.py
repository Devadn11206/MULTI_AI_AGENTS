from fastapi import APIRouter, HTTPException
from api.schemas.result import WorkflowResultResponse
from api.services.workflow_service import WorkflowService

router = APIRouter(prefix="/workflows", tags=["results"])

@router.get("/{workflow_id}/result", response_model=WorkflowResultResponse)
def get_workflow_result(workflow_id: str):
    res = WorkflowService.get_result(workflow_id)
    if not res:
        raise HTTPException(status_code=404, detail="Result not found or workflow not completed.")
    return res
