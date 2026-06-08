"""
Task 7 — Reranking Module.

Phương pháp sử dụng:
  1. Cross-encoder: Jina Reranker v2 (multilingual) qua API  — độ chính xác cao nhất.
  2. RRF (Reciprocal Rank Fusion): gộp nhiều ranked-list, không cần model.
  3. MMR (Maximal Marginal Relevance): tăng diversity, giảm trùng lặp.

`rerank()` mặc định dùng cross-encoder (Jina), fallback RRF nếu API không khả dụng.
"""

import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Cross-encoder via Jina Reranker v2 API
# ---------------------------------------------------------------------------

def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank bằng Jina Reranker v2-base-multilingual (cross-encoder).

    Cross-encoder: đưa (query, doc) vào model cùng lúc → score chính xác hơn
    bi-encoder vì tận dụng attention cross giữa query và document.

    Fallback về sort theo original score nếu API không khả dụng.
    """
    if not candidates:
        return []

    jina_key = os.getenv("JINA_API_KEY", "")
    if jina_key:
        try:
            import requests as req

            resp = req.post(
                "https://api.jina.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {jina_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k,
                },
                timeout=20,
            )
            if resp.status_code == 200:
                results = []
                for r in resp.json()["results"]:
                    item = candidates[r["index"]].copy()
                    item["score"] = float(r["relevance_score"])
                    results.append(item)
                return results
        except Exception:
            pass

    # Fallback: sort theo original score
    sorted_cands = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    return sorted_cands[:top_k]


# ---------------------------------------------------------------------------
# RRF — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker (Cormack et al. 2009).

    RRF(d) = Σ_r  1 / (k + rank_r(d))

    k=60: hằng số smoothing, làm giảm ảnh hưởng của các tài liệu ở top,
    giúp tài liệu xuất hiện đều ở nhiều list nhận điểm cao.
    Phù hợp để merge dense + sparse results mà không cần normalize score.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# MMR — Maximal Marginal Relevance
# ---------------------------------------------------------------------------

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — cân bằng giữa relevance và diversity.

    MMR(d) = λ · sim(query, d) − (1−λ) · max_{s ∈ S} sim(d, s)

    lambda_param=0.7: ưu tiên relevance 70%, diversity 30%.
    Giảm redundancy khi nhiều chunks từ cùng 1 văn bản pháp luật.
    """
    if not candidates:
        return []

    # Cần embedding trong mỗi candidate; nếu không có thì lấy từ index
    def _get_emb(c: dict) -> np.ndarray | None:
        if "embedding" in c:
            return np.array(c["embedding"], dtype=np.float32)
        return None

    q_vec = np.array(query_embedding, dtype=np.float32)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)

    # Tính cosine sim (query, candidate)
    relevances = []
    embs = []
    for c in candidates:
        emb = _get_emb(c)
        if emb is not None:
            norm_emb = emb / (np.linalg.norm(emb) + 1e-9)
            relevances.append(float(q_vec @ norm_emb))
            embs.append(norm_emb)
        else:
            # Dùng original score nếu không có embedding
            relevances.append(c.get("score", 0.0))
            embs.append(None)

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx, best_score = None, float("-inf")
        for idx in remaining:
            relevance = relevances[idx]
            max_sim = 0.0
            for sel_idx in selected_indices:
                if embs[idx] is not None and embs[sel_idx] is not None:
                    sim = float(embs[idx] @ embs[sel_idx])
                    max_sim = max(max_sim, sim)
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, idx

        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for sel_idx in selected_indices:
        item = candidates[sel_idx].copy()
        item["score"] = relevances[sel_idx]
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.

    method:
      - "cross_encoder": Jina API, fallback sort-by-score (mặc định)
      - "rrf": Reciprocal Rank Fusion (cần dùng rerank_rrf trực tiếp với ranked_lists)
      - "mmr": Maximal Marginal Relevance (cần query_embedding)
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "rrf":
        # RRF hoạt động trên nhiều list — ở đây xử lý 1 list như single-ranker
        return rerank_rrf([candidates], top_k=top_k)
    elif method == "mmr":
        from src.task4_chunking_indexing import _get_embedding_model
        model = _get_embedding_model()
        q_emb = model.encode(query, normalize_embeddings=True).tolist()
        return rerank_mmr(q_emb, candidates, top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2–7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
