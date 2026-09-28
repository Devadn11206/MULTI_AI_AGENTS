from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from core.database import get_db
from evaluation.schemas import EvaluationRunSchema, EvaluationRunCreate, EvaluationResultCreate
from evaluation.repository import EvaluationRepository

router = APIRouter(prefix="/evaluations", tags=["evaluations"])

@router.get("", response_model=List[EvaluationRunSchema])
def list_evaluations(limit: int = 100, skip: int = 0, db: Session = Depends(get_db)):
    repo = EvaluationRepository(db)
    return repo.get_runs(limit=limit, skip=skip)

@router.get("/{run_id}", response_model=EvaluationRunSchema)
def get_evaluation(run_id: str, db: Session = Depends(get_db)):
    repo = EvaluationRepository(db)
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run

# NOTE: The actual execution of evaluations is handled by the CLI / background worker,
# but we could expose a POST /evaluations/run endpoint if we wanted to trigger it via UI.
