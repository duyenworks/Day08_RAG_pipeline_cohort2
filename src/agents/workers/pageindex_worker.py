"""Worker using external capability through PageIndex (Task 8)."""

from __future__ import annotations

from src.agents.messages import WorkerRequest, WorkerResponse
from src.task8_pageindex_vectorless import pageindex_search


def run_pageindex_worker(request: WorkerRequest) -> WorkerResponse:
    top_k = int(request.context.get("top_k", 5))
    chunks = pageindex_search(query=request.query, top_k=top_k)

    if not chunks:
        return WorkerResponse(
            worker="pageindex_worker",
            status="skipped",
            payload={"chunks": [], "retrieval_source": "none"},
            reasoning="PageIndex returned no results.",
        )

    best_score = chunks[0]["score"]
    return WorkerResponse(
        worker="pageindex_worker",
        status="ok",
        payload={
            "chunks": chunks,
            "best_score": best_score,
            "retrieval_source": "pageindex",
            "external_capability": "pageindex_api",
        },
        reasoning=f"PageIndex returned {len(chunks)} chunks, best_score={best_score:.4f}.",
    )
