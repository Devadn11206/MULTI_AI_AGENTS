"""
Phase 15 — Observability: TraceRepository

Low-level persistence layer. Uses existing PostgreSQL DB from Phase 14.
Agents MUST NOT call this directly — use TraceManager.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy.orm import Session

from observability.db_model import TraceEventRecord
from observability.models import TraceEvent

logger = logging.getLogger(__name__)


class TraceRepository:
    """Handles raw persistence and retrieval of trace events."""

    @staticmethod
    def save_event(event: TraceEvent) -> None:
        """Persist a single TraceEvent. Failures are swallowed — never interrupt workflow."""
        from core.database import SessionLocal
        db: Session = SessionLocal()
        try:
            record = TraceEventRecord(
                id              = event.event_id,
                event_id        = event.event_id,
                workflow_id     = event.workflow_id,
                trace_id        = event.trace_id,
                parent_event_id = event.parent_event_id,
                timestamp       = event.timestamp,
                agent_name      = event.agent_name,
                event_type      = event.event_type.value,
                status          = event.status.value,
                message         = (event.message or "")[:500],
                action          = event.action,
                decision        = event.decision,
                tool_name       = event.tool_name,
                result_summary  = (event.result_summary or "")[:300],
                duration_ms     = event.duration_ms,
                iteration       = event.iteration,
                provider        = event.provider,
                model           = event.model,
                error_code      = event.error_code,
                event_metadata  = event.metadata,
            )
            db.add(record)
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.warning(f"[TraceRepository] Could not save event {event.event_id}: {exc}")
        finally:
            db.close()

    @staticmethod
    def get_events(
        db: Session,
        workflow_id: str,
        *,
        agent_name:  Optional[str] = None,
        event_type:  Optional[str] = None,
        status:      Optional[str] = None,
        start_time:  Optional[datetime] = None,
        end_time:    Optional[datetime] = None,
        limit:       int = 100,
        offset:      int = 0,
    ) -> List[TraceEventRecord]:
        """Retrieve trace events with filtering and pagination."""
        limit = min(limit, 500)  # hard cap
        q = db.query(TraceEventRecord).filter(
            TraceEventRecord.workflow_id == workflow_id
        )
        if agent_name:
            q = q.filter(TraceEventRecord.agent_name == agent_name)
        if event_type:
            q = q.filter(TraceEventRecord.event_type == event_type)
        if status:
            q = q.filter(TraceEventRecord.status == status)
        if start_time:
            q = q.filter(TraceEventRecord.timestamp >= start_time)
        if end_time:
            q = q.filter(TraceEventRecord.timestamp <= end_time)
        return (
            q.order_by(TraceEventRecord.timestamp.asc())
             .offset(offset)
             .limit(limit)
             .all()
        )

    @staticmethod
    def count_events(db: Session, workflow_id: str) -> int:
        return db.query(TraceEventRecord).filter(
            TraceEventRecord.workflow_id == workflow_id
        ).count()

    @staticmethod
    def get_trace_id(db: Session, workflow_id: str) -> Optional[str]:
        record = (
            db.query(TraceEventRecord.trace_id)
              .filter(TraceEventRecord.workflow_id == workflow_id)
              .first()
        )
        return record.trace_id if record else None

    @staticmethod
    def cleanup_old_traces(db: Session, retention_days: int) -> int:
        """Delete trace events older than retention_days. Returns count deleted."""
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        deleted = (
            db.query(TraceEventRecord)
              .filter(TraceEventRecord.created_at < cutoff)
              .delete(synchronize_session=False)
        )
        db.commit()
        return deleted
