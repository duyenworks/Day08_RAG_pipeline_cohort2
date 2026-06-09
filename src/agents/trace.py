"""Trace helpers for observable reasoning flow."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from src.agents.state import PipelineState, TraceEvent


def now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def append_trace(
    state: PipelineState,
    *,
    step: str,
    agent: str,
    input_summary: str,
    output_summary: str,
    duration_ms: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append a structured trace event to shared state."""
    state.trace.append(
        TraceEvent(
            ts=now_iso(),
            step=step,
            agent=agent,
            input_summary=input_summary,
            output_summary=output_summary,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
    )


class StepTimer:
    """Tiny helper for measuring worker/supervisor step durations."""

    def __init__(self) -> None:
        self._start = perf_counter()

    def elapsed_ms(self) -> int:
        return int((perf_counter() - self._start) * 1000)


def format_trace_timeline(state: PipelineState) -> str:
    """Render human-readable trace timeline for demo/CLI."""
    if not state.trace:
        return "No trace events."
    lines = []
    for idx, event in enumerate(state.trace, 1):
        lines.append(
            f"{idx:02d}. [{event.agent}] {event.step} "
            f"({event.duration_ms}ms) - {event.output_summary}"
        )
    return "\n".join(lines)
