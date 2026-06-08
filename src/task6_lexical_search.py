"""
Task 6 — Lexical Search Module (BM25).

BM25: TF × IDF với length normalization (k1=1.5, b=0.75).
Phù hợp truy vấn từ khóa chính xác (số điều luật, tên chất cấm).
"""

import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.task4_chunking_indexing import load_index

CORPUS: list[dict] = []
_BM25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản: lowercase + tách theo khoảng trắng/ký tự đặc biệt."""
    text = text.lower()
    return [t for t in re.split(r"[\s\W_]+", text) if t]


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """Xây dựng BM25 index từ corpus."""
    tokenized = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized)


def _ensure_index():
    """Lazy-load BM25 index từ local chunks."""
    global CORPUS, _BM25
    if _BM25 is not None:
        return

    CORPUS, _ = load_index()
    _BM25 = build_bm25_index(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    try:
        _ensure_index()
    except FileNotFoundError:
        return []

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = _BM25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append({
            "content": CORPUS[idx]["content"],
            "score": score,
            "metadata": CORPUS[idx].get("metadata", {}),
        })
    return results


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
