"""
Phase 15 — Observability: Trace Event Models

Defines the structured data types for agent tracing.
No chain-of-thought is exposed — only decisions, actions, and results.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class AgentEventType(str, Enum):
    # Workflow lifecycle
    WORKFLOW_STARTED     = "WORKFLOW_STARTED"
    WORKFLOW_COMPLETED   = "WORKFLOW_COMPLETED"
    WORKFLOW_FAILED      = "WORKFLOW_FAILED"

    # Agent lifecycle
    AGENT_STARTED        = "AGENT_STARTED"
    AGENT_COMPLETED      = "AGENT_COMPLETED"
    AGENT_FAILED         = "AGENT_FAILED"

    # Decisions (safe summaries — no chain-of-thought)
    DECISION             = "DECISION"

    # Tool calls
    TOOL_STARTED         = "TOOL_STARTED"
    TOOL_COMPLETED       = "TOOL_COMPLETED"
    TOOL_FAILED          = "TOOL_FAILED"

    # State transitions
    STATE_UPDATED        = "STATE_UPDATED"
    HANDOFF              = "HANDOFF"

    # LLM calls
    LLM_REQUEST          = "LLM_REQUEST"
    LLM_RESPONSE         = "LLM_RESPONSE"
    LLM_RETRY            = "LLM_RETRY"
    LLM_FALLBACK         = "LLM_FALLBACK"

    # Checkpointing (Phase 13 integration)
    CHECKPOINT_CREATED   = "CHECKPOINT_CREATED"
    CHECKPOINT_RESTORED  = "CHECKPOINT_RESTORED"

    # Training / Evaluation
    TRAINING_STARTED     = "TRAINING_STARTED"
    TRAINING_COMPLETED   = "TRAINING_COMPLETED"
    EVALUATION_STARTED   = "EVALUATION_STARTED"
    EVALUATION_COMPLETED = "EVALUATION_COMPLETED"
    REQUIREMENT_PASSED   = "REQUIREMENT_PASSED"
    REQUIREMENT_FAILED   = "REQUIREMENT_FAILED"

    # Debugger / Self-correction
    DEBUGGER_ANALYSIS    = "DEBUGGER_ANALYSIS"
    RETRAINING_STARTED   = "RETRAINING_STARTED"

    # Review
    REVIEW_STARTED       = "REVIEW_STARTED"
    REVIEW_COMPLETED     = "REVIEW_COMPLETED"

    # Generic error
    ERROR                = "ERROR"


class TraceStatus(str, Enum):
    RUNNING   = "RUNNING"
    SUCCESS   = "SUCCESS"
    FAILED    = "FAILED"
    WARNING   = "WARNING"
    SKIPPED   = "SKIPPED"
    RETRYING  = "RETRYING"
    FALLBACK  = "FALLBACK"


# ─── Critical event set (never dropped from buffer) ──────────────────────────

CRITICAL_EVENT_TYPES = {
    AgentEventType.WORKFLOW_FAILED,
    AgentEventType.WORKFLOW_COMPLETED,
    AgentEventType.AGENT_FAILED,
    AgentEventType.REQUIREMENT_FAILED,
    AgentEventType.REQUIREMENT_PASSED,
    AgentEventType.REVIEW_COMPLETED,
    AgentEventType.CHECKPOINT_CREATED,
    AgentEventType.CHECKPOINT_RESTORED,
}


# ─── Core Trace Event ─────────────────────────────────────────────────────────

class TraceEvent(BaseModel):
    """
    A single observable event in the MLForge agent trace.

    IMPORTANT: Never include chain-of-thought, raw prompts, API keys, or secrets.
    Only include safe, structured decision/action/result metadata.
    """
    event_id:       str = Field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:12]}")
    workflow_id:    str
    trace_id:       str
    parent_event_id: Optional[str] = None

    timestamp:      datetime = Field(default_factory=datetime.utcnow)

    agent_name:     str
    event_type:     AgentEventType
    status:         TraceStatus

    # Safe human-readable summary — no chain-of-thought
    message:        str = ""

    # Structured metadata (safe)
    action:         Optional[str] = None          # e.g. "model_selected"
    decision:       Optional[str] = None          # e.g. "selected_candidate_models"
    tool_name:      Optional[str] = None          # e.g. "dataset_reader"
    result_summary: Optional[str] = None          # e.g. "Dataset: 500 rows, 11 cols"
    duration_ms:    Optional[int] = None
    iteration:      Optional[int] = None

    # LLM metadata (no prompts/responses)
    provider:       Optional[str] = None          # "gemini" / "groq"
    model:          Optional[str] = None

    # Error metadata (no stack traces)
    error_code:     Optional[str] = None

    # Arbitrary safe JSON metadata (no secrets, no datasets, no prompts)
    metadata:       Optional[Dict[str, Any]] = None

    @property
    def is_critical(self) -> bool:
        return self.event_type in CRITICAL_EVENT_TYPES


# ─── Trace Summary ────────────────────────────────────────────────────────────

class TraceSummary(BaseModel):
    workflow_id:    str
    trace_id:       str
    status:         str
    total_events:   int
    agents:         int
    retries:        int
    fallbacks:      int
    failures:       int
    duration_ms:    Optional[int] = None
    agent_durations: Dict[str, Optional[int]] = Field(default_factory=dict)
    tool_usage:     Dict[str, int] = Field(default_factory=dict)
    provider_usage: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    decisions:      list = Field(default_factory=list)
