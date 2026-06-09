"""Message contract between supervisor and workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

WorkerTask = Literal["retrieve", "pageindex_fallback", "generate"]
WorkerStatus = Literal["ok", "error", "skipped"]


@dataclass
class WorkerRequest:
    """A minimal and explicit contract sent by supervisor to workers."""

    task: WorkerTask
    query: str
    trace_id: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerResponse:
    """Standardized worker response for traceability and composition."""

    worker: str
    status: WorkerStatus
    payload: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    trace_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SupervisorDecision:
    """Supervisor routing decision and state patch plan."""

    next_worker: str | None
    reason: str
    merged_state_patch: dict[str, Any] = field(default_factory=dict)
