"""Tests for supervisor-worker architecture and trace observability."""

from __future__ import annotations

from src.agents.messages import WorkerRequest, WorkerResponse
from src.agents.run import run_pipeline
from src.agents.state import PipelineState
from src.agents.supervisor import RAGSupervisor


def test_message_contract_minimum_fields() -> None:
    req = WorkerRequest(task="retrieve", query="test", trace_id="t1", context={"top_k": 3})
    resp = WorkerResponse(worker="retrieval_worker", status="ok", payload={"chunks": []}, reasoning="ok")
    assert req.task == "retrieve"
    assert req.trace_id == "t1"
    assert resp.worker == "retrieval_worker"
    assert isinstance(resp.payload, dict)


def test_pipeline_state_has_trace_field() -> None:
    state = PipelineState(query="q", trace_id="trace-1")
    assert isinstance(state.trace, list)
    assert state.status == "running"


def test_supervisor_decision_pageindex_when_score_low() -> None:
    sup = RAGSupervisor(score_threshold=0.5)
    decision = sup._decide_after_retrieval(best_score=0.01, retrieval_source="hybrid")
    assert decision.next_worker == "pageindex_worker"
    assert "fallback" in decision.reason.lower()


def test_supervisor_decision_skip_pageindex_when_score_high() -> None:
    sup = RAGSupervisor(score_threshold=0.005)
    decision = sup._decide_after_retrieval(best_score=0.2, retrieval_source="hybrid")
    assert decision.next_worker is None
    assert "skip" in decision.reason.lower()


def test_run_pipeline_returns_observable_trace() -> None:
    result = run_pipeline("Hình phạt tàng trữ ma túy?", top_k=3, use_reranking=False)
    assert result.status in {"completed", "failed"}
    assert len(result.trace) >= 3  # supervisor.start + retrieval + generation (and maybe fallback)
    steps = [event.step for event in result.trace]
    assert "supervisor.start" in steps
    assert "worker.retrieval" in steps
    assert "worker.generate" in steps
