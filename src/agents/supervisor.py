"""Supervisor that coordinates retrieval, external fallback, and generation workers."""

from __future__ import annotations

from src.agents.messages import SupervisorDecision, WorkerRequest
from src.agents.state import PipelineState
from src.agents.trace import StepTimer, append_trace
from src.agents.workers import (
    run_generation_worker,
    run_pageindex_worker,
    run_retrieval_worker,
)


class RAGSupervisor:
    """Rule-based supervisor for a transparent multi-worker flow."""

    def __init__(self, *, score_threshold: float = 0.005) -> None:
        self.score_threshold = score_threshold

    def run(self, state: PipelineState, *, top_k: int = 5, use_reranking: bool = True) -> PipelineState:
        self._trace_supervisor_start(state, top_k=top_k, use_reranking=use_reranking)

        # 1) Retrieval worker
        retrieve_req = WorkerRequest(
            task="retrieve",
            query=state.query,
            trace_id=state.trace_id,
            context={
                "top_k": top_k,
                "use_reranking": use_reranking,
                "score_threshold": self.score_threshold,
            },
        )
        retrieval_resp = self._run_retrieval(state, retrieve_req)
        chunks = retrieval_resp.payload.get("chunks", [])
        best_score = float(retrieval_resp.payload.get("best_score", 0.0))
        retrieval_source = str(retrieval_resp.payload.get("retrieval_source", "none"))

        decision = self._decide_after_retrieval(best_score=best_score, retrieval_source=retrieval_source)
        state.supervisor_plan.append(decision.reason)
        append_trace(
            state,
            step="supervisor.decide",
            agent="supervisor",
            input_summary=f"best_score={best_score:.4f}, source={retrieval_source}",
            output_summary=decision.reason,
            duration_ms=0,
            metadata={"next_worker": decision.next_worker},
        )

        # 2) Optional external capability worker (PageIndex)
        if decision.next_worker == "pageindex_worker":
            pageindex_req = WorkerRequest(
                task="pageindex_fallback",
                query=state.query,
                trace_id=state.trace_id,
                context={"top_k": top_k},
            )
            pageindex_resp = self._run_pageindex(state, pageindex_req)
            pageindex_chunks = pageindex_resp.payload.get("chunks", [])
            if pageindex_chunks:
                chunks = pageindex_chunks
                retrieval_source = "pageindex"

        # 3) Generation worker
        generate_req = WorkerRequest(
            task="generate",
            query=state.query,
            trace_id=state.trace_id,
            context={"chunks": chunks, "retrieval_source": retrieval_source},
        )
        self._run_generation(state, generate_req)

        return state

    def _trace_supervisor_start(self, state: PipelineState, *, top_k: int, use_reranking: bool) -> None:
        append_trace(
            state,
            step="supervisor.start",
            agent="supervisor",
            input_summary=state.query[:120],
            output_summary="Initialized plan and dispatched retrieval worker.",
            duration_ms=0,
            metadata={"top_k": top_k, "use_reranking": use_reranking},
        )

    def _run_retrieval(self, state: PipelineState, request: WorkerRequest):
        timer = StepTimer()
        resp = run_retrieval_worker(request)
        state.worker_outputs[resp.worker] = {
            "status": resp.status,
            "reasoning": resp.reasoning,
            "payload": resp.payload,
        }
        state.retrieved_chunks = list(resp.payload.get("chunks", []))
        state.retrieval_source = str(resp.payload.get("retrieval_source", "none"))
        append_trace(
            state,
            step="worker.retrieval",
            agent=resp.worker,
            input_summary=f"top_k={request.context.get('top_k', 5)}",
            output_summary=resp.reasoning,
            duration_ms=timer.elapsed_ms(),
            metadata={"status": resp.status},
        )
        return resp

    def _run_pageindex(self, state: PipelineState, request: WorkerRequest):
        timer = StepTimer()
        resp = run_pageindex_worker(request)
        state.worker_outputs[resp.worker] = {
            "status": resp.status,
            "reasoning": resp.reasoning,
            "payload": resp.payload,
        }
        chunks = resp.payload.get("chunks", [])
        if chunks:
            state.retrieved_chunks = list(chunks)
            state.retrieval_source = "pageindex"
        append_trace(
            state,
            step="worker.pageindex_fallback",
            agent=resp.worker,
            input_summary=f"top_k={request.context.get('top_k', 5)}",
            output_summary=resp.reasoning,
            duration_ms=timer.elapsed_ms(),
            metadata={"status": resp.status, "external_capability": "pageindex_api"},
        )
        return resp

    def _run_generation(self, state: PipelineState, request: WorkerRequest):
        timer = StepTimer()
        resp = run_generation_worker(request)
        state.worker_outputs[resp.worker] = {
            "status": resp.status,
            "reasoning": resp.reasoning,
            "payload": resp.payload,
        }
        state.answer = str(resp.payload.get("answer", ""))
        state.sources = list(resp.payload.get("sources", []))
        state.retrieval_source = str(resp.payload.get("retrieval_source", state.retrieval_source))
        state.status = "completed" if resp.status in ("ok", "skipped") else "failed"
        if resp.status == "error":
            state.error = resp.reasoning
        append_trace(
            state,
            step="worker.generate",
            agent=resp.worker,
            input_summary=f"chunks={len(request.context.get('chunks', []))}",
            output_summary=resp.reasoning,
            duration_ms=timer.elapsed_ms(),
            metadata={"status": resp.status},
        )
        return resp

    def _decide_after_retrieval(self, *, best_score: float, retrieval_source: str) -> SupervisorDecision:
        if retrieval_source == "pageindex":
            return SupervisorDecision(
                next_worker=None,
                reason="Hybrid pipeline already switched to PageIndex fallback internally.",
                merged_state_patch={},
            )
        if best_score < self.score_threshold:
            return SupervisorDecision(
                next_worker="pageindex_worker",
                reason=(
                    "Best hybrid score below threshold; run PageIndex worker "
                    "for external capability fallback."
                ),
                merged_state_patch={"needs_pageindex_fallback": True},
            )
        return SupervisorDecision(
            next_worker=None,
            reason="Hybrid score is good enough; skip PageIndex worker.",
            merged_state_patch={"needs_pageindex_fallback": False},
        )
