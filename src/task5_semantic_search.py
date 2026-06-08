"""
Task 5 — Semantic Search Module.

Dense retrieval trên vector store (Weaviate Cloud hoặc local cosine similarity).
"""

import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.task4_chunking_indexing import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    _get_embedding_model,
    load_index,
)

load_dotenv()


def _search_local(query: str, top_k: int) -> list[dict]:
    """Cosine similarity trên local index (embeddings đã normalize)."""
    chunks, embeddings = load_index()
    model = _get_embedding_model()
    query_vec = model.encode(query, normalize_embeddings=True)

    scores = embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        chunk = chunks[idx]
        results.append({
            "content": chunk["content"],
            "score": score,
            "metadata": chunk.get("metadata", {}),
        })
    return results


def _search_weaviate(query: str, top_k: int) -> list[dict]:
    """Query Weaviate Cloud bằng near_vector."""
    import weaviate
    from weaviate.classes.init import Auth
    from weaviate.classes.query import MetadataQuery

    url = os.getenv("WEAVIATE_URL", "")
    api_key = os.getenv("WEAVIATE_API_KEY", "")
    if not url.startswith("http"):
        url = f"https://{url}"

    model = _get_embedding_model()
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key),
    )
    try:
        collection = client.collections.get(COLLECTION_NAME)
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for obj in response.objects:
            distance = obj.metadata.distance if obj.metadata else 1.0
            results.append({
                "content": obj.properties.get("content", ""),
                "score": 1.0 - distance,
                "metadata": {
                    "source": obj.properties.get("source", ""),
                    "type": obj.properties.get("doc_type", ""),
                    "chunk_index": obj.properties.get("chunk_index", 0),
                },
            })
        return results
    finally:
        client.close()


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    try:
        if os.getenv("WEAVIATE_URL") and os.getenv("WEAVIATE_API_KEY"):
            try:
                results = _search_weaviate(query, top_k)
                if results:
                    return results
            except Exception:
                pass
        return _search_local(query, top_k)
    except FileNotFoundError:
        return []


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
