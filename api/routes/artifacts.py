from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os

router = APIRouter(prefix="/artifacts", tags=["artifacts"])

@router.get("/{artifact_id}/download")
def download_artifact(artifact_id: str):
    # This is a stub for secure artifact download
    # In a real system, we look up artifact_id in a DB to get the safe file path.
    # We never accept raw paths.
    raise HTTPException(status_code=404, detail="Artifact downloads not fully implemented in this phase.")

@router.get("/workflows/{workflow_id}/artifacts")
def list_workflow_artifacts(workflow_id: str):
    # Stub for listing artifacts
    return {"artifacts": []}
