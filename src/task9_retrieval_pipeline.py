"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic + lexical search → RRF merge → cross-encoder rerank
→ PageIndex fallback nếu score < threshold.

Logic:
    Query
      ├→ Semantic Search (dense)  ─┐
      ├→ Lexical Search  (BM25)  ──┤→ RRF Merge → Rerank → Results
      └→ Nếu best_score < threshold → Fallback PageIndex
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank, rerank_rrf
from src.task8_pageindex_vectorless import pageindex_search

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Ngưỡng điểm RRF tối thiểu sau rerank; dưới ngưỡng → fallback PageIndex.
# RRF score điển hình: 0.01–0.05 với k=60. Chọn 0.005 để chỉ fallback
# khi thực sự không tìm được kết quả liên quan.
SCORE_THRESHOLD = 0.005
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # Jina API, tự fallback nếu không có network


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': str}
        source = 'hybrid' | 'pageindex'
    """
    # Step 1: Dense + Sparse retrieval (lấy nhiều hơn để merge tốt)
    fetch_k = max(top_k * 3, 15)
    dense_results = semantic_search(query, top_k=fetch_k)
    sparse_results = lexical_search(query, top_k=fetch_k)

    # Step 2: Merge bằng RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=fetch_k)
    for item in merged:
        item["source"] = "hybrid"

    if not merged:
        # Không có kết quả nào → thẳng fallback
        return _fallback_pageindex(query, top_k)

    # Step 3: Rerank
    if use_reranking and len(merged) > 1:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        except Exception:
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    # Step 4: Fallback nếu best score < threshold
    best_score = final_results[0]["score"] if final_results else 0.0
    if best_score < score_threshold:
        fallback = _fallback_pageindex(query, top_k)
        if fallback:
            return fallback
        # PageIndex cũng thất bại → trả về hybrid dù score thấp
    return final_results[:top_k]


def _fallback_pageindex(query: str, top_k: int) -> list[dict]:
    """Gọi PageIndex fallback, không crash nếu không khả dụng."""
    try:
        results = pageindex_search(query, top_k=top_k)
        return results if results else []
    except Exception:
        return []


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì liên quan tới ma tuý năm 2024",
        "Luật phòng chống ma tuý 2025 quy định gì về cai nghiện",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] [{r['source']}] {r['content'][:80]}...")
