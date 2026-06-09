"""Shared state schema for supervisor-worker orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    """A single observable event emitted by supervisor or workers."""

    ts: str
    step: str
    agent: str
    input_summary: str
    output_summary: str
    duration_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState:
    """Shared mutable state passed across supervisor and workers."""

    query: str
    trace_id: str
    status: str = "running"
    supervisor_plan: list[str] = field(default_factory=list)
    worker_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    messages: list[dict[str, str]] = field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieval_source: str = "none"
    trace: list[TraceEvent] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict for UI and CLI rendering."""
        return {
            "query": self.query,
            "trace_id": self.trace_id,
            "status": self.status,
            "supervisor_plan": list(self.supervisor_plan),
            "worker_outputs": dict(self.worker_outputs),
            "messages": list(self.messages),
            "retrieved_chunks": list(self.retrieved_chunks),
            "answer": self.answer,
            "sources": list(self.sources),
            "retrieval_source": self.retrieval_source,
            "trace": [
                {
                    "ts": event.ts,
                    "step": event.step,
                    "agent": event.agent,
                    "input_summary": event.input_summary,
                    "output_summary": event.output_summary,
                    "duration_ms": event.duration_ms,
                    "metadata": event.metadata,
                }
                for event in self.trace
            ],
            "error": self.error,
        }
