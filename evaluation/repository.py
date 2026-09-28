from sqlalchemy.orm import Session
from typing import List, Optional
import datetime
from evaluation.db_models import EvaluationRun, EvaluationResult
from evaluation.schemas import EvaluationRunCreate, EvaluationResultCreate

class EvaluationRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_run(self, run_create: EvaluationRunCreate) -> EvaluationRun:
        run = EvaluationRun(
            evaluation_run_id=run_create.evaluation_run_id,
            mode=run_create.mode,
            total_tests=run_create.total_tests,
            passed_tests=run_create.passed_tests,
            failed_tests=run_create.failed_tests,
            duration_ms=run_create.duration_ms,
            summary=run_create.summary,
            started_at=datetime.datetime.utcnow()
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def update_run(self, run_id: str, updates: dict) -> Optional[EvaluationRun]:
        run = self.get_run(run_id)
        if not run:
            return None
        for k, v in updates.items():
            setattr(run, k, v)
        self.db.commit()
        self.db.refresh(run)
        return run

    def get_run(self, run_id: str) -> Optional[EvaluationRun]:
        return self.db.query(EvaluationRun).filter(EvaluationRun.evaluation_run_id == run_id).first()

    def get_runs(self, limit: int = 100, skip: int = 0) -> List[EvaluationRun]:
        return self.db.query(EvaluationRun).order_by(EvaluationRun.started_at.desc()).offset(skip).limit(limit).all()

    def add_result(self, result_create: EvaluationResultCreate) -> EvaluationResult:
        res = EvaluationResult(**result_create.model_dump())
        self.db.add(res)
        self.db.commit()
        self.db.refresh(res)
        return res

    def get_results(self, run_id: str) -> List[EvaluationResult]:
        return self.db.query(EvaluationResult).filter(EvaluationResult.evaluation_run_id == run_id).all()
