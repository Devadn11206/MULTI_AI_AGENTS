"""
Phase 15 — Observability: FastAPI Trace Routes

GET /workflows/{workflow_id}/trace          — paginated trace events
GET /workflows/{workflow_id}/trace/summary  — aggregated statistics
GET /workflows/{workflow_id}/trace/export   — JSON export

Security: Only the workflow owner can access its trace (Phase 14 authorization).
"""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from api.dependencies.auth import get_current_user
from api.services.workflow_service import WorkflowService
from core.database import SessionLocal
from models.domain import User
from observability.service import TraceService

router = APIRouter(prefix="/workflows", tags=["trace"])


def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{workflow_id}/trace")
def get_trace(
    workflow_id: str,
    agent:       Optional[str] = Query(None, description="Filter by agent name"),
    event_type:  Optional[str] = Query(None, description="Filter by event type"),
    status:      Optional[str] = Query(None, description="Filter by status"),
    start_time:  Optional[datetime] = Query(None),
    end_time:    Optional[datetime] = Query(None),
    limit:       int = Query(100, ge=1, le=500),
    offset:      int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db = Depends(_get_db),
):
    """Return chronologically ordered trace events for a workflow."""
    WorkflowService.authorize_workflow(workflow_id, current_user.id)
    events = TraceService.get_trace(
        db, workflow_id,
        agent_name=agent, event_type=event_type, status=status,
        start_time=start_time, end_time=end_time,
        limit=limit, offset=offset,
    )
    return {
        "workflow_id": workflow_id,
        "total":       len(events),
        "offset":      offset,
        "limit":       limit,
        "events":      events,
    }


@router.get("/{workflow_id}/trace/summary")
def get_trace_summary(
    workflow_id:  str,
    current_user: User = Depends(get_current_user),
    db = Depends(_get_db),
):
    """Return aggregated statistics for the workflow trace."""
    WorkflowService.authorize_workflow(workflow_id, current_user.id)
    summary = TraceService.get_summary(db, workflow_id)
    if not summary:
        raise HTTPException(status_code=404, detail="No trace found for this workflow.")
    return summary.model_dump()


@router.get("/{workflow_id}/trace/export")
def export_trace(
    workflow_id:  str,
    current_user: User = Depends(get_current_user),
    db = Depends(_get_db),
):
    """Export full workflow trace as downloadable JSON (no secrets or prompts)."""
    WorkflowService.authorize_workflow(workflow_id, current_user.id)
    json_data = TraceService.export_trace_json(db, workflow_id)
    return Response(
        content=json_data,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=trace_{workflow_id}.json"},
    )
