"""Entry points for running supervisor + workers pipeline."""

from __future__ import annotations

import uuid

from src.agents.state import PipelineState
from src.agents.supervisor import RAGSupervisor
from src.agents.trace import format_trace_timeline


def run_pipeline(query: str, *, top_k: int = 5, use_reranking: bool = True) -> PipelineState:
    """Run full supervisor-worker flow and return final shared state."""
    state = PipelineState(query=query, trace_id=str(uuid.uuid4()))
    supervisor = RAGSupervisor()
    return supervisor.run(state, top_k=top_k, use_reranking=use_reranking)


def run_pipeline_dict(query: str, *, top_k: int = 5, use_reranking: bool = True) -> dict:
    """Compatibility helper for UI code expecting plain dict."""
    return run_pipeline(query, top_k=top_k, use_reranking=use_reranking).to_dict()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG supervisor-worker pipeline")
    parser.add_argument("query", type=str, help="User query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    result = run_pipeline(args.query, top_k=args.top_k, use_reranking=not args.no_rerank)
    print("=" * 70)
    print(f"Trace ID: {result.trace_id}")
    print(f"Status:   {result.status}")
    print(f"Source:   {result.retrieval_source}")
    print("-" * 70)
    print("ANSWER:")
    print(result.answer)
    print("-" * 70)
    print("TRACE:")
    print(format_trace_timeline(result))
