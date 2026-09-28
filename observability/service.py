"""
Phase 15 — Observability: TraceService

Service layer for trace retrieval, aggregation, and statistics.
Enforces authorization via workflow ownership (reuses Phase 14 auth).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from observability.db_model import TraceEventRecord
from observability.models import AgentEventType, TraceSummary
from observability.repository import TraceRepository

logger = logging.getLogger(__name__)


class TraceService:
    """High-level service for trace querying and analysis."""

    @staticmethod
    def get_trace(
        db: Session,
        workflow_id: str,
        *,
        agent_name:  Optional[str]      = None,
        event_type:  Optional[str]      = None,
        status:      Optional[str]      = None,
        start_time:  Optional[datetime] = None,
        end_time:    Optional[datetime] = None,
        limit:       int = 100,
        offset:      int = 0,
    ) -> List[Dict[str, Any]]:
        """Return ordered trace events as JSON-serializable dicts."""
        records = TraceRepository.get_events(
            db, workflow_id,
            agent_name=agent_name, event_type=event_type, status=status,
            start_time=start_time, end_time=end_time,
            limit=limit, offset=offset,
        )
        return [TraceService._record_to_dict(r) for r in records]

    @staticmethod
    def get_summary(db: Session, workflow_id: str) -> Optional[TraceSummary]:
        """Aggregate statistics across all trace events for a workflow."""
        records = TraceRepository.get_events(db, workflow_id, limit=500)
        if not records:
            return None

        trace_id    = records[0].trace_id if records else ""
        agents      = set(r.agent_name for r in records)
        retries     = sum(1 for r in records if r.event_type == AgentEventType.LLM_RETRY.value)
        fallbacks   = sum(1 for r in records if r.event_type == AgentEventType.LLM_FALLBACK.value)
        failures    = sum(1 for r in records if r.status == "FAILED")

        # Workflow duration from first to last event
        wf_start  = records[0].timestamp
        wf_end    = records[-1].timestamp
        duration  = int((wf_end - wf_start).total_seconds() * 1000) if wf_start and wf_end else None

        # Per-agent durations: sum duration_ms of AGENT_COMPLETED events
        agent_durations: Dict[str, Optional[int]] = {}
        for r in records:
            if r.event_type == AgentEventType.AGENT_COMPLETED.value and r.duration_ms:
                agent_durations[r.agent_name] = (
                    (agent_durations.get(r.agent_name) or 0) + r.duration_ms
                )

        # Tool usage count
        tool_usage: Dict[str, int] = {}
        for r in records:
            if r.event_type == AgentEventType.TOOL_COMPLETED.value and r.tool_name:
                tool_usage[r.tool_name] = tool_usage.get(r.tool_name, 0) + 1

        # Provider usage
        provider_usage: Dict[str, Dict[str, int]] = {}
        for r in records:
            if r.provider:
                p = r.provider
                if p not in provider_usage:
                    provider_usage[p] = {"calls": 0, "successes": 0, "failures": 0, "fallbacks": 0}
                if r.event_type == AgentEventType.LLM_RESPONSE.value:
                    provider_usage[p]["calls"]     += 1
                    provider_usage[p]["successes"] += 1
                elif r.event_type == AgentEventType.LLM_FALLBACK.value:
                    provider_usage[p]["fallbacks"] += 1
                    provider_usage[p]["calls"]     += 1

        # Decisions
        decisions = [
            {"agent": r.agent_name, "decision": r.decision, "message": r.message}
            for r in records
            if r.event_type == AgentEventType.DECISION.value and r.decision
        ]

        # Overall status
        final = next(
            (r for r in reversed(records)
             if r.event_type in (AgentEventType.WORKFLOW_COMPLETED.value, AgentEventType.WORKFLOW_FAILED.value)),
            None
        )
        wf_status = (final.status if final else "RUNNING")

        return TraceSummary(
            workflow_id    = workflow_id,
            trace_id       = trace_id,
            status         = wf_status,
            total_events   = len(records),
            agents         = len(agents),
            retries        = retries,
            fallbacks      = fallbacks,
            failures       = failures,
            duration_ms    = duration,
            agent_durations = agent_durations,
            tool_usage     = tool_usage,
            provider_usage = provider_usage,
            decisions      = decisions,
        )

    @staticmethod
    def export_trace_json(db: Session, workflow_id: str) -> str:
        """Export complete trace as a JSON string. Safe: no secrets or chain-of-thought."""
        records = TraceRepository.get_events(db, workflow_id, limit=500)
        data    = [TraceService._record_to_dict(r) for r in records]
        return json.dumps({"workflow_id": workflow_id, "events": data}, indent=2, default=str)

    @staticmethod
    def _record_to_dict(r: TraceEventRecord) -> Dict[str, Any]:
        return {
            "event_id":        r.event_id,
            "workflow_id":     r.workflow_id,
            "trace_id":        r.trace_id,
            "parent_event_id": r.parent_event_id,
            "timestamp":       r.timestamp.isoformat() if r.timestamp else None,
            "agent_name":      r.agent_name,
            "event_type":      r.event_type,
            "status":          r.status,
            "message":         r.message,
            "action":          r.action,
            "decision":        r.decision,
            "tool_name":       r.tool_name,
            "result_summary":  r.result_summary,
            "duration_ms":     r.duration_ms,
            "iteration":       r.iteration,
            "provider":        r.provider,
            "model":           r.model,
            "error_code":      r.error_code,
            "metadata":        r.event_metadata,
        }
