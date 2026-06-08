"""
Task 10 — Generation Có Citation.

Pipeline:
  1. Retrieve chunks (Task 9)
  2. Reorder để tránh "Lost in the Middle" (Liu et al. 2023)
  3. Format context với source label cho citation
  4. Call OpenAI GPT-4o-mini với SYSTEM_PROMPT có citation rules
  5. Return {'answer', 'sources', 'retrieval_source'}
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv()

from src.task9_retrieval_pipeline import retrieve

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# top_k=5: đủ evidence (3 legal + 2 news) mà không quá dài → tránh lost-in-middle.
TOP_K = 5

# top_p=0.9: nucleus sampling, đủ diverse nhưng không hallucinate.
TOP_P = 0.9

# temperature=0.2: RAG cần factual output, ít sáng tạo. Thấp hơn 0.3
# vì văn bản pháp luật yêu cầu chính xác tuyệt đối.
TEMPERATURE = 0.2

LLM_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Trả lời câu hỏi bằng tiếng Việt một cách đầy đủ và chính xác.

Với mỗi thông tin hoặc tuyên bố sự kiện, hãy ngay lập tức chèn citation trong ngoặc vuông
liên kết đến nguồn cụ thể (ví dụ: [Luật Phòng chống ma tuý 2025, Điều 3]
hoặc [VnExpress, 2024]).

Quy tắc BẮT BUỘC:
- Chỉ sử dụng thông tin từ context được cung cấp
- Mỗi tuyên bố sự kiện PHẢI có citation
- Nếu context không đủ thông tin, trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
- Không suy đoán hoặc bịa đặt thông tin
- Cấu trúc câu trả lời với các đoạn văn rõ ràng"""

# ---------------------------------------------------------------------------
# Document reordering (Liu et al. 2023 — Lost in the Middle)
# ---------------------------------------------------------------------------

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "Lost in the Middle".

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI, bỏ sót thông tin ở GIỮA.
    Strategy: xen kẽ — chunk quan trọng lẻ vị trí vào đầu,
    chunk quan trọng chẵn vị trí vào cuối (đảo ngược).

    Input  (sorted desc): [0, 1, 2, 3, 4]  (0 = quan trọng nhất)
    Output:               [0, 2, 4, 3, 1]
    """
    if len(chunks) <= 2:
        return chunks

    # Chunk ở vị trí chẵn (0-indexed) → đầu list
    head = [chunks[i] for i in range(0, len(chunks), 2)]
    # Chunk ở vị trí lẻ → cuối list (đảo ngược để chunk quan trọng nhất ở cuối)
    tail = [chunks[i] for i in range(len(chunks) - 1 - (len(chunks) % 2 == 0), 0, -2)]

    return head + tail


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string với source label cho citation.

    Mỗi chunk được đánh số [Document N | source | type]
    để LLM biết cite đúng nguồn.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")
        # Rút gọn tên file cho citation dễ đọc
        source_label = source.replace(".md", "").replace(".docx", "").replace(".pdf", "")
        parts.append(
            f"[Document {i} | Nguồn: {source_label} | Loại: {doc_type}]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Chunks đã dùng
            'retrieval_source': str  # 'hybrid' | 'pageindex' | 'none'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    retrieval_source = chunks[0].get("source", "hybrid")

    # Step 2: Reorder để tránh lost in the middle
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}"

    # Step 5: Call LLM
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có (thiếu OPENAI_API_KEY).",
            "sources": chunks,
            "retrieval_source": retrieval_source,
        }

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = response.choices[0].message.content or ""
    except Exception as e:
        answer = f"Tôi không thể xác minh thông tin này từ nguồn hiện có. (Lỗi LLM: {e})"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    questions = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2025?",
    ]
    for q in questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
