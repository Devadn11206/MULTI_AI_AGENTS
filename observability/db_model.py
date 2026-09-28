"""
Phase 15 — Observability: SQLAlchemy Domain Model for trace_events table.
Added to the existing domain.py models — reuses the same PostgreSQL DB.
"""
from sqlalchemy import Column, String, Integer, DateTime, Text, Index, JSON
from datetime import datetime
from core.database import Base


class TraceEventRecord(Base):
    """
    Persistent storage for agent trace events.

    Retention: governed by TRACE_RETENTION_DAYS setting.
    Ownership: tied to workflow_id which is owned by a user (authorization via workflow).

    IMPORTANT: Never store chain-of-thought, raw prompts, API keys, or secrets here.
    """
    __tablename__ = "trace_events"

    id               = Column(String(36),  primary_key=True)    # event_id
    event_id         = Column(String(36),  nullable=False, unique=True)
    workflow_id      = Column(String(50),  nullable=False, index=True)
    trace_id         = Column(String(50),  nullable=False, index=True)
    parent_event_id  = Column(String(36),  nullable=True)

    timestamp        = Column(DateTime,    nullable=False, default=datetime.utcnow, index=True)

    agent_name       = Column(String(100), nullable=False, index=True)
    event_type       = Column(String(50),  nullable=False, index=True)
    status           = Column(String(30),  nullable=False, index=True)

    # Human-readable summaries (bounded length)
    message          = Column(String(500), nullable=True)
    action           = Column(String(200), nullable=True)
    decision         = Column(String(200), nullable=True)
    tool_name        = Column(String(100), nullable=True)
    result_summary   = Column(String(300), nullable=True)

    # Performance
    duration_ms      = Column(Integer,     nullable=True)
    iteration        = Column(Integer,     nullable=True)

    # LLM metadata (no prompts/responses)
    provider         = Column(String(50),  nullable=True)
    model            = Column(String(100), nullable=True)

    # Error metadata (no stack traces)
    error_code       = Column(String(100), nullable=True)

    # Arbitrary safe JSON (bounded)
    event_metadata   = Column(JSON,        nullable=True)

    created_at       = Column(DateTime,    nullable=False, default=datetime.utcnow)

    # Compound indexes for efficient trace retrieval
    __table_args__ = (
        Index("ix_trace_wf_ts",    "workflow_id", "timestamp"),
        Index("ix_trace_wf_agent", "workflow_id", "agent_name"),
        Index("ix_trace_wf_type",  "workflow_id", "event_type"),
        Index("ix_trace_id_ts",    "trace_id",    "timestamp"),
    )
