"""
Task 8 — PageIndex Vectorless RAG.

PageIndex hiểu cấu trúc document (tree/outline) thay vì embedding vectors.
Phù hợp cho văn bản pháp luật có cấu trúc chương/điều/khoản rõ ràng.

Luồng:
  1. upload_documents(): upload DOCX lên PageIndex, lưu doc_ids vào data/index/pageindex_docs.json
  2. pageindex_search(): gửi query, poll kết quả, trả về list chunks

Đăng ký tại: https://pageindex.ai/
SDK: https://github.com/VectifyAI/PageIndex
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
DOC_IDS_FILE = INDEX_DIR / "pageindex_docs.json"


def _get_client():
    from pageindex.client import PageIndexClient
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY chưa được cấu hình trong .env")
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents(force: bool = False) -> dict[str, str]:
    """
    Upload toàn bộ DOCX lên PageIndex. Lưu mapping filename → doc_id.
    Bỏ qua file đã upload trước đó (trừ khi force=True).

    Returns: {"filename": "doc_id", ...}
    """
    client = _get_client()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing doc_ids
    existing: dict[str, str] = {}
    if DOC_IDS_FILE.exists() and not force:
        existing = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))

    legal_dir = LANDING_DIR / "legal"
    updated = dict(existing)

    for docx_file in sorted(legal_dir.glob("*.docx")):
        if docx_file.name in existing:
            print(f"  ⊘ Đã upload: {docx_file.name}")
            continue
        print(f"  ↑ Uploading: {docx_file.name}")
        try:
            result = client.submit_document(str(docx_file))
            doc_id = result.get("doc_id") or result.get("id", "")
            if doc_id:
                updated[docx_file.name] = doc_id
                print(f"    ✓ doc_id={doc_id}")
        except Exception as e:
            print(f"    ✗ Lỗi: {e}")

    DOC_IDS_FILE.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  ✓ Lưu doc_ids → {DOC_IDS_FILE}")
    return updated


def _load_doc_ids() -> dict[str, str]:
    """Load doc_ids đã lưu từ lần upload trước."""
    if not DOC_IDS_FILE.exists():
        return {}
    return json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))


def _poll_retrieval(client, retrieval_id: str, timeout: int = 30) -> dict:
    """Poll retrieval result cho đến khi sẵn sàng hoặc timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = client.get_retrieval(retrieval_id)
        status = result.get("status", "")
        if status in ("completed", "done", "success") or "results" in result or "nodes" in result:
            return result
        if status in ("failed", "error"):
            return {}
        time.sleep(1.5)
    return {}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    try:
        client = _get_client()
        doc_ids = _load_doc_ids()

        if not doc_ids:
            # Thử upload lần đầu nếu chưa có
            doc_ids = upload_documents()

        if not doc_ids:
            return []

        results: list[dict] = []
        rank = 1

        for filename, doc_id in list(doc_ids.items())[:3]:  # Query tối đa 3 docs
            try:
                # Submit query
                submit_resp = client.submit_query(doc_id=doc_id, query=query)
                retrieval_id = submit_resp.get("retrieval_id") or submit_resp.get("id", "")
                if not retrieval_id:
                    continue

                # Poll for results
                retrieval_resp = _poll_retrieval(client, retrieval_id, timeout=25)
                if not retrieval_resp:
                    continue

                # Parse results — PageIndex trả về nodes/results list
                nodes = (
                    retrieval_resp.get("nodes")
                    or retrieval_resp.get("results")
                    or retrieval_resp.get("data")
                    or []
                )

                for node in nodes[:top_k]:
                    content = (
                        node.get("content")
                        or node.get("text")
                        or node.get("chunk")
                        or str(node)
                    )
                    score = node.get("score", 1.0 / rank)
                    results.append({
                        "content": content,
                        "score": float(score),
                        "metadata": {
                            "source": filename,
                            "type": "legal",
                            "doc_id": doc_id,
                        },
                        "source": "pageindex",
                    })
                    rank += 1

            except Exception:
                continue

        # Sort + deduplicate
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    except Exception:
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
    else:
        print("Uploading documents...")
        upload_documents()
        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
