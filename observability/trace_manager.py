"""
Phase 15 — Observability: TraceManager

The single entry point for all agent trace instrumentation.
Agents call TraceManager methods — they never write SQL directly.

Design: fire-and-forget with a bounded in-memory buffer.
If DB persistence fails, the ML workflow is NOT interrupted.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from observability.models import (
    AgentEventType, TraceEvent, TraceStatus, CRITICAL_EVENT_TYPES
)

logger = logging.getLogger(__name__)

# ─── Buffer limits ────────────────────────────────────────────────────────────
_BUFFER_MAX      = 500   # Maximum pending events in memory
_BUFFER_DROP_LOW = 100   # How many low-priority events to drop when buffer full


class TraceManager:
    """
    Central trace manager for MLForge observability.

    - Agents call high-level methods (start_event, record_decision, etc.)
    - Events are written to an in-memory buffer and persisted to DB asynchronously
    - If persistence fails, the ML workflow continues (observability is secondary)
    """

    def __init__(self, workflow_id: str, trace_id: Optional[str] = None):
        self.workflow_id = workflow_id
        self.trace_id    = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
        self._buffer: deque[TraceEvent] = deque(maxlen=_BUFFER_MAX)
        self._start_times: Dict[str, float] = {}  # event_id → start wall time
        self._agent_start_times: Dict[str, float] = {}  # agent_name → start
        self._trace_start: float = time.monotonic()

    # ─── Low-level event factory ──────────────────────────────────────────────

    def _make_event(
        self, *,
        agent_name:      str,
        event_type:      AgentEventType,
        status:          TraceStatus,
        message:         str        = "",
        action:          Optional[str]  = None,
        decision:        Optional[str]  = None,
        tool_name:       Optional[str]  = None,
        result_summary:  Optional[str]  = None,
        duration_ms:     Optional[int]  = None,
        iteration:       Optional[int]  = None,
        provider:        Optional[str]  = None,
        model:           Optional[str]  = None,
        error_code:      Optional[str]  = None,
        metadata:        Optional[Dict[str, Any]] = None,
        parent_event_id: Optional[str]  = None,
    ) -> TraceEvent:
        return TraceEvent(
            workflow_id     = self.workflow_id,
            trace_id        = self.trace_id,
            parent_event_id = parent_event_id,
            agent_name      = agent_name,
            event_type      = event_type,
            status          = status,
            message         = message[:500] if message else "",   # bounded
            action          = action,
            decision        = decision,
            tool_name       = tool_name,
            result_summary  = (result_summary or "")[:300],
            duration_ms     = duration_ms,
            iteration       = iteration,
            provider        = provider,
            model           = model,
            error_code      = error_code,
            metadata        = metadata,
        )

    def _emit(self, event: TraceEvent) -> None:
        """Add event to buffer and persist asynchronously."""
        self._buffer_event(event)
        try:
            self._persist(event)
        except Exception as exc:
            # Observability MUST NOT crash the ML workflow.
            logger.warning(f"[TraceManager] _emit persistence failed: {exc}")

    def _buffer_event(self, event: TraceEvent) -> None:
        """Add to bounded in-memory buffer. Drop non-critical events when full."""
        if len(self._buffer) >= _BUFFER_MAX:
            # Drop low-priority events from the front of the buffer
            dropped = 0
            new_buf: deque[TraceEvent] = deque(maxlen=_BUFFER_MAX)
            for ev in self._buffer:
                if ev.is_critical or dropped >= _BUFFER_DROP_LOW:
                    new_buf.append(ev)
                else:
                    dropped += 1
            self._buffer = new_buf
            logger.warning(f"[TraceManager] Buffer full — dropped {dropped} low-priority events.")
        self._buffer.append(event)

    def _persist(self, event: TraceEvent) -> None:
        """Persist to PostgreSQL. Failures are logged but never propagate."""
        try:
            from observability.repository import TraceRepository
            TraceRepository.save_event(event)
        except Exception as exc:
            # IMPORTANT: Do NOT re-raise. Trace failure must not stop the workflow.
            logger.warning(f"[TraceManager] Trace persistence failed: {exc}")

    # ─── High-level API ───────────────────────────────────────────────────────

    def start_workflow(self, message: str = "Workflow started.") -> TraceEvent:
        ev = self._make_event(
            agent_name  = "orchestrator",
            event_type  = AgentEventType.WORKFLOW_STARTED,
            status      = TraceStatus.RUNNING,
            message     = message,
        )
        self._trace_start = time.monotonic()
        self._emit(ev)
        return ev

    def complete_workflow(self, message: str = "Workflow completed.", status: TraceStatus = TraceStatus.SUCCESS) -> TraceEvent:
        duration = int((time.monotonic() - self._trace_start) * 1000)
        ev = self._make_event(
            agent_name  = "orchestrator",
            event_type  = AgentEventType.WORKFLOW_COMPLETED,
            status      = status,
            message     = message,
            duration_ms = duration,
        )
        self._emit(ev)
        return ev

    def fail_workflow(self, message: str, error_code: Optional[str] = None) -> TraceEvent:
        duration = int((time.monotonic() - self._trace_start) * 1000)
        ev = self._make_event(
            agent_name  = "orchestrator",
            event_type  = AgentEventType.WORKFLOW_FAILED,
            status      = TraceStatus.FAILED,
            message     = message[:500],
            error_code  = error_code,
            duration_ms = duration,
        )
        self._emit(ev)
        return ev

    def start_agent(self, agent_name: str, message: str = "", iteration: Optional[int] = None) -> TraceEvent:
        self._agent_start_times[agent_name] = time.monotonic()
        ev = self._make_event(
            agent_name = agent_name,
            event_type = AgentEventType.AGENT_STARTED,
            status     = TraceStatus.RUNNING,
            message    = message or f"{agent_name} started.",
            iteration  = iteration,
        )
        self._emit(ev)
        return ev

    def complete_agent(self, agent_name: str, message: str = "", result_summary: Optional[str] = None) -> TraceEvent:
        start = self._agent_start_times.pop(agent_name, None)
        duration = int((time.monotonic() - start) * 1000) if start else None
        ev = self._make_event(
            agent_name     = agent_name,
            event_type     = AgentEventType.AGENT_COMPLETED,
            status         = TraceStatus.SUCCESS,
            message        = message or f"{agent_name} completed.",
            result_summary = result_summary,
            duration_ms    = duration,
        )
        self._emit(ev)
        return ev

    def fail_agent(self, agent_name: str, message: str, error_code: Optional[str] = None) -> TraceEvent:
        start = self._agent_start_times.pop(agent_name, None)
        duration = int((time.monotonic() - start) * 1000) if start else None
        ev = self._make_event(
            agent_name  = agent_name,
            event_type  = AgentEventType.AGENT_FAILED,
            status      = TraceStatus.FAILED,
            message     = message[:500],
            error_code  = error_code,
            duration_ms = duration,
        )
        self._emit(ev)
        return ev

    def record_decision(
        self, agent_name: str, decision: str, message: str,
        action: Optional[str]     = None,
        metadata: Optional[Dict]  = None,
        iteration: Optional[int]  = None,
    ) -> TraceEvent:
        ev = self._make_event(
            agent_name = agent_name,
            event_type = AgentEventType.DECISION,
            status     = TraceStatus.SUCCESS,
            message    = message[:500],
            decision   = decision,
            action     = action,
            metadata   = metadata,
            iteration  = iteration,
        )
        self._emit(ev)
        return ev

    def start_tool(self, agent_name: str, tool_name: str, message: str = "") -> str:
        """Returns event_id to pass to complete_tool / fail_tool."""
        ev = self._make_event(
            agent_name = agent_name,
            event_type = AgentEventType.TOOL_STARTED,
            status     = TraceStatus.RUNNING,
            message    = message or f"Tool {tool_name} started.",
            tool_name  = tool_name,
        )
        self._start_times[ev.event_id] = time.monotonic()
        self._emit(ev)
        return ev.event_id

    def complete_tool(self, agent_name: str, tool_name: str, result_summary: str, parent_event_id: Optional[str] = None) -> TraceEvent:
        start = self._start_times.pop(parent_event_id, None)
        duration = int((time.monotonic() - start) * 1000) if start else None
        ev = self._make_event(
            agent_name      = agent_name,
            event_type      = AgentEventType.TOOL_COMPLETED,
            status          = TraceStatus.SUCCESS,
            message         = f"Tool {tool_name} completed.",
            tool_name       = tool_name,
            result_summary  = result_summary[:300],
            duration_ms     = duration,
            parent_event_id = parent_event_id,
        )
        self._emit(ev)
        return ev

    def fail_tool(self, agent_name: str, tool_name: str, message: str, error_code: Optional[str] = None, parent_event_id: Optional[str] = None) -> TraceEvent:
        start = self._start_times.pop(parent_event_id, None)
        duration = int((time.monotonic() - start) * 1000) if start else None
        ev = self._make_event(
            agent_name      = agent_name,
            event_type      = AgentEventType.TOOL_FAILED,
            status          = TraceStatus.FAILED,
            message         = message[:500],
            tool_name       = tool_name,
            error_code      = error_code,
            duration_ms     = duration,
            parent_event_id = parent_event_id,
        )
        self._emit(ev)
        return ev

    def record_handoff(self, from_agent: str, to_agent: str, message: str = "") -> TraceEvent:
        ev = self._make_event(
            agent_name = from_agent,
            event_type = AgentEventType.HANDOFF,
            status     = TraceStatus.SUCCESS,
            message    = message or f"Handed off to {to_agent}.",
            metadata   = {"target_agent": to_agent},
        )
        self._emit(ev)
        return ev

    def record_state_change(self, agent_name: str, message: str, metadata: Optional[Dict] = None) -> TraceEvent:
        ev = self._make_event(
            agent_name = agent_name,
            event_type = AgentEventType.STATE_UPDATED,
            status     = TraceStatus.SUCCESS,
            message    = message[:500],
            metadata   = metadata,
        )
        self._emit(ev)
        return ev

    def record_llm_request(self, agent_name: str, provider: str, model: str, metadata: Optional[Dict] = None) -> str:
        ev = self._make_event(
            agent_name = agent_name,
            event_type = AgentEventType.LLM_REQUEST,
            status     = TraceStatus.RUNNING,
            message    = f"LLM request sent via {provider}.",
            provider   = provider,
            model      = model,
            metadata   = metadata,
        )
        self._start_times[ev.event_id] = time.monotonic()
        self._emit(ev)
        return ev.event_id

    def record_llm_response(self, agent_name: str, provider: str, model: str, parent_event_id: Optional[str] = None, metadata: Optional[Dict] = None) -> TraceEvent:
        start = self._start_times.pop(parent_event_id, None)
        duration = int((time.monotonic() - start) * 1000) if start else None
        ev = self._make_event(
            agent_name      = agent_name,
            event_type      = AgentEventType.LLM_RESPONSE,
            status          = TraceStatus.SUCCESS,
            message         = f"LLM response received from {provider}.",
            provider        = provider,
            model           = model,
            duration_ms     = duration,
            metadata        = metadata,
            parent_event_id = parent_event_id,
        )
        self._emit(ev)
        return ev

    def record_llm_retry(self, agent_name: str, provider: str, attempt: int, reason: str = "") -> TraceEvent:
        ev = self._make_event(
            agent_name = agent_name,
            event_type = AgentEventType.LLM_RETRY,
            status     = TraceStatus.RETRYING,
            message    = f"LLM retry #{attempt} on {provider}. {reason}".strip(),
            provider   = provider,
            iteration  = attempt,
        )
        self._emit(ev)
        return ev

    def record_llm_fallback(self, agent_name: str, from_provider: str, to_provider: str, reason: str = "") -> TraceEvent:
        ev = self._make_event(
            agent_name = agent_name,
            event_type = AgentEventType.LLM_FALLBACK,
            status     = TraceStatus.FALLBACK,
            message    = f"Falling back from {from_provider} to {to_provider}. {reason}".strip(),
            provider   = to_provider,
            metadata   = {"from_provider": from_provider, "to_provider": to_provider},
        )
        self._emit(ev)
        return ev

    def record_checkpoint(self, phase: str, restored: bool = False) -> TraceEvent:
        ev = self._make_event(
            agent_name = "orchestrator",
            event_type = AgentEventType.CHECKPOINT_RESTORED if restored else AgentEventType.CHECKPOINT_CREATED,
            status     = TraceStatus.SUCCESS,
            message    = (
                f"Workflow restored from checkpoint at {phase}." if restored
                else f"Checkpoint created after {phase}."
            ),
            metadata   = {"phase": phase},
        )
        self._emit(ev)
        return ev

    def record_training(self, agent_name: str, started: bool, model_name: Optional[str] = None,
                        duration_ms: Optional[int] = None, metadata: Optional[Dict] = None, iteration: Optional[int] = None) -> TraceEvent:
        ev = self._make_event(
            agent_name  = agent_name,
            event_type  = AgentEventType.TRAINING_STARTED if started else AgentEventType.TRAINING_COMPLETED,
            status      = TraceStatus.RUNNING if started else TraceStatus.SUCCESS,
            message     = (f"Training started for {model_name}." if started else f"Training completed for {model_name}."),
            duration_ms = duration_ms,
            metadata    = metadata or ({"model_name": model_name} if model_name else {}),
            iteration   = iteration,
        )
        self._emit(ev)
        return ev

    def record_evaluation(self, agent_name: str, started: bool, metadata: Optional[Dict] = None, iteration: Optional[int] = None) -> TraceEvent:
        ev = self._make_event(
            agent_name = agent_name,
            event_type = AgentEventType.EVALUATION_STARTED if started else AgentEventType.EVALUATION_COMPLETED,
            status     = TraceStatus.RUNNING if started else TraceStatus.SUCCESS,
            message    = "Evaluation started." if started else "Evaluation completed.",
            metadata   = metadata,
            iteration  = iteration,
        )
        self._emit(ev)
        return ev

    def record_requirement(self, passed: bool, message: str, metadata: Optional[Dict] = None, iteration: Optional[int] = None) -> TraceEvent:
        ev = self._make_event(
            agent_name = "evaluation",
            event_type = AgentEventType.REQUIREMENT_PASSED if passed else AgentEventType.REQUIREMENT_FAILED,
            status     = TraceStatus.SUCCESS if passed else TraceStatus.WARNING,
            message    = message[:500],
            metadata   = metadata,
            iteration  = iteration,
        )
        self._emit(ev)
        return ev

    def record_debugger(self, message: str, decision: Optional[str] = None, metadata: Optional[Dict] = None, iteration: Optional[int] = None) -> TraceEvent:
        ev = self._make_event(
            agent_name = "debugger",
            event_type = AgentEventType.DEBUGGER_ANALYSIS,
            status     = TraceStatus.SUCCESS,
            message    = message[:500],
            decision   = decision,
            metadata   = metadata,
            iteration  = iteration,
        )
        self._emit(ev)
        return ev

    def record_review(self, started: bool, passed: Optional[bool] = None, message: str = "", metadata: Optional[Dict] = None) -> TraceEvent:
        if started:
            ev_type, status = AgentEventType.REVIEW_STARTED, TraceStatus.RUNNING
            msg = "Review started."
        else:
            ev_type = AgentEventType.REVIEW_COMPLETED
            status  = TraceStatus.SUCCESS if passed else TraceStatus.WARNING
            msg     = message or ("Reviewer approved the solution." if passed else "Reviewer identified issues.")
        ev = self._make_event(
            agent_name = "reviewer",
            event_type = ev_type,
            status     = status,
            message    = msg[:500],
            metadata   = metadata,
        )
        self._emit(ev)
        return ev

    def record_error(self, agent_name: str, message: str, error_code: Optional[str] = None, metadata: Optional[Dict] = None) -> TraceEvent:
        ev = self._make_event(
            agent_name = agent_name,
            event_type = AgentEventType.ERROR,
            status     = TraceStatus.FAILED,
            message    = message[:500],
            error_code = error_code,
            metadata   = metadata,
        )
        self._emit(ev)
        return ev

    # ─── Buffer introspection (for testing) ──────────────────────────────────

    def get_buffered_events(self) -> List[TraceEvent]:
        return list(self._buffer)

    def buffer_size(self) -> int:
        return len(self._buffer)


# ─── Global registry: one TraceManager per active workflow ───────────────────

_managers: Dict[str, TraceManager] = {}


def get_trace_manager(workflow_id: str) -> Optional[TraceManager]:
    return _managers.get(workflow_id)


def create_trace_manager(workflow_id: str, trace_id: Optional[str] = None) -> TraceManager:
    tm = TraceManager(workflow_id=workflow_id, trace_id=trace_id)
    _managers[workflow_id] = tm
    return tm


def remove_trace_manager(workflow_id: str) -> None:
    _managers.pop(workflow_id, None)
