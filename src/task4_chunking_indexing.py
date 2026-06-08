"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
META_FILE = INDEX_DIR / "meta.json"

# =============================================================================
# CONFIGURATION
# =============================================================================

# RecursiveCharacterTextSplitter: an toàn với cả văn bản pháp luật dài và bài báo;
# tách theo đoạn (\n\n) → câu → từ, giữ ngữ cảnh tốt hơn split cố định.
CHUNK_SIZE = 500
# Overlap 10% giúp không mất mạch logic giữa các điều/khoản liền kề.
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# all-MiniLM-L6-v2: 80MB, dim=384, encode CPU ~10× nhanh hơn bge-m3.
# Đủ chất lượng cho tiếng Việt trong scope bài lab (corpus < 2000 chunks).
# Nếu cần độ chính xác cao hơn cho production, đổi sang BAAI/bge-m3.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Weaviate Cloud (nếu có .env); luôn lưu cache local để search offline.
VECTOR_STORE = "weaviate"
COLLECTION_NAME = "DrugLawDocs"

_MODEL = None


def _get_embedding_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        # local_files_only=True: dùng cache, không check HuggingFace khi offline/sandbox
        try:
            _MODEL = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        except Exception:
            _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def load_documents() -> list[dict]:
    """Đọc toàn bộ markdown files từ data/standardized/."""
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        rel = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = rel.parts[0] if rel.parts else "unknown"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "path": str(rel),
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents bằng RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            if not chunk_text.strip():
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed chunks bằng BAAI/bge-m3."""
    model = _get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=len(texts) > 50, normalize_embeddings=True)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def _save_local_index(chunks: list[dict]):
    """Lưu chunks + embeddings ra disk để Task 5/6 dùng offline."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    serializable = [
        {"content": c["content"], "metadata": c["metadata"]}
        for c in chunks
    ]
    CHUNKS_FILE.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    np.save(EMBEDDINGS_FILE, embeddings)

    META_FILE.write_text(
        json.dumps({
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunking_method": CHUNKING_METHOD,
            "num_chunks": len(chunks),
        }, indent=2),
        encoding="utf-8",
    )


def _index_to_weaviate(chunks: list[dict]):
    """Index chunks lên Weaviate Cloud (nếu có credentials)."""
    url = os.getenv("WEAVIATE_URL", "")
    api_key = os.getenv("WEAVIATE_API_KEY", "")
    if not url or not api_key:
        print("  ⊘ Bỏ qua Weaviate (chưa cấu hình WEAVIATE_URL / WEAVIATE_API_KEY)")
        return

    import weaviate
    from weaviate.classes.config import Configure, DataType, Property
    from weaviate.classes.init import Auth

    if not url.startswith("http"):
        url = f"https://{url}"

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key),
    )
    try:
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)

        collection = client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )

        with collection.batch.fixed_size(batch_size=50) as batch:
            for chunk in chunks:
                meta = chunk["metadata"]
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": meta.get("source", ""),
                        "doc_type": meta.get("type", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                    },
                    vector=chunk["embedding"],
                )

        print(f"  ✓ Indexed {len(chunks)} chunks → Weaviate '{COLLECTION_NAME}'")
    finally:
        client.close()


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks vào local cache + Weaviate (nếu có)."""
    _save_local_index(chunks)
    print(f"  ✓ Saved local index → {INDEX_DIR}")

    if VECTOR_STORE == "weaviate":
        try:
            _index_to_weaviate(chunks)
        except Exception as e:
            print(f"  ⚠ Weaviate indexing failed (local index vẫn dùng được): {e}")


def load_index() -> tuple[list[dict], np.ndarray]:
    """Load chunks và embeddings từ local index (dùng cho Task 5/6)."""
    if not CHUNKS_FILE.exists() or not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            "Index chưa tồn tại. Chạy: python src/task4_chunking_indexing.py"
        )
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_FILE)
    return chunks, embeddings


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexing complete")


if __name__ == "__main__":
    run_pipeline()
