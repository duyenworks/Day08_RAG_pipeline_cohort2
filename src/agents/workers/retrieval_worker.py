"""Worker that runs hybrid retrieval pipeline (Task 9)."""

from __future__ import annotations

from src.agents.messages import WorkerRequest, WorkerResponse
from src.task9_retrieval_pipeline import retrieve


def run_retrieval_worker(request: WorkerRequest) -> WorkerResponse:
    top_k = int(request.context.get("top_k", 5))
    use_reranking = bool(request.context.get("use_reranking", True))
    score_threshold = float(request.context.get("score_threshold", 0.005))

    chunks = retrieve(
        query=request.query,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranking=use_reranking,
    )
    best_score = chunks[0]["score"] if chunks else 0.0
    retrieval_source = chunks[0].get("source", "none") if chunks else "none"
    reasoning = (
        f"Retrieved {len(chunks)} chunks, best_score={best_score:.4f}, "
        f"source={retrieval_source}."
    )
    return WorkerResponse(
        worker="retrieval_worker",
        status="ok",
        payload={
            "chunks": chunks,
            "best_score": best_score,
            "retrieval_source": retrieval_source,
            "used_pageindex_internally": retrieval_source == "pageindex",
        },
        reasoning=reasoning,
    )
