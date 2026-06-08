"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


_LEGAL_META = {
    "luat-phong-chong-ma-tuy-2025": {
        "title": "Luật Phòng, chống ma túy 2025 (Luật số 47/2025/QH15)",
        "source": "https://vanban.chinhphu.vn",
        "issued_by": "Quốc hội nước CHXHCN Việt Nam",
        "date": "2025-06-17",
        "description": (
            "Luật quy định về phòng ngừa, ngăn chặn, đấu tranh chống tệ nạn ma tuý; "
            "kiểm soát các hoạt động hợp pháp liên quan đến ma tuý; "
            "trách nhiệm của cá nhân, gia đình, cơ quan, tổ chức và Nhà nước "
            "trong phòng, chống ma tuý."
        ),
    },
    "nghi-dinh-1632026ND-CP-huong-dan-luat-phong-chong-ma-tuy": {
        "title": "Nghị định 163/2026/NĐ-CP hướng dẫn thi hành Luật Phòng, chống ma túy",
        "source": "https://vanban.chinhphu.vn",
        "issued_by": "Chính phủ nước CHXHCN Việt Nam",
        "date": "2026-01-15",
        "description": (
            "Nghị định quy định chi tiết và hướng dẫn thi hành một số điều của "
            "Luật Phòng, chống ma túy 2025 về quản lý người sử dụng trái phép "
            "chất ma tuý, cai nghiện ma tuý và quản lý sau cai nghiện."
        ),
    },
    "nghi-dinh-28-2026-ND-CP-quy-dinh-cac-danh-muc-chat-ma-tuy-va-tien-chat": {
        "title": "Nghị định 28/2026/NĐ-CP quy định các danh mục chất ma túy và tiền chất",
        "source": "https://vanban.chinhphu.vn",
        "issued_by": "Chính phủ nước CHXHCN Việt Nam",
        "date": "2026-01-08",
        "description": (
            "Nghị định quy định danh mục chất ma tuý, tiền chất và thuốc gây nghiện, "
            "hướng thần, thuốc tiên chất dùng làm thuốc nghiện, thuốc hướng thần."
        ),
    },
}


def _make_scanned_placeholder(filepath: Path, meta: dict) -> str:
    """Tạo nội dung markdown cho PDF bị quét (không có text layer)."""
    return (
        f"# {meta['title']}\n\n"
        f"**Cơ quan ban hành:** {meta['issued_by']}\n"
        f"**Ngày ban hành:** {meta['date']}\n"
        f"**Nguồn:** {meta['source']}\n"
        f"**File gốc:** `{filepath.name}`\n\n"
        "---\n\n"
        "## Mô tả\n\n"
        f"{meta['description']}\n\n"
        "---\n\n"
        "> **Lưu ý kỹ thuật:** File PDF gốc là bản quét (scanned image), "
        "không có text layer. Nội dung đầy đủ cần OCR để trích xuất.\n"
    )


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    converted = 0
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            text = result.text_content.strip()

            output_path = output_dir / f"{filepath.stem}.md"
            if text:
                output_path.write_text(text, encoding="utf-8")
                print(f"  ✓ Saved (text): {output_path} ({len(text):,} chars)")
            else:
                # Scanned PDF — dùng placeholder với metadata
                meta = _LEGAL_META.get(filepath.stem, {
                    "title": filepath.stem.replace("-", " ").title(),
                    "source": "N/A",
                    "issued_by": "N/A",
                    "date": "N/A",
                    "description": "Văn bản pháp luật về phòng chống ma tuý.",
                })
                placeholder = _make_scanned_placeholder(filepath, meta)
                output_path.write_text(placeholder, encoding="utf-8")
                print(f"  ✓ Saved (placeholder): {output_path} ({len(placeholder):,} chars)")
            converted += 1

    print(f"  → Đã convert {converted} file pháp luật")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            content = header + data.get("content_markdown", "")
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path} ({len(content):,} chars)")
            converted += 1

    print(f"  → Đã convert {converted} bài báo")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
