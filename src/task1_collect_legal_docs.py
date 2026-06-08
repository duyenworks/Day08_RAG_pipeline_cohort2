"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Danh mục văn bản pháp luật về ma tuý (tối thiểu 3 file)
LEGAL_DOCUMENTS = [
    {
        "filename": "luat-phong-chong-ma-tuy-2025.docx",
        "title": "Luật Phòng, chống ma túy 2025 (Luật số 47/2025/QH15)",
        "source": "https://vanban.chinhphu.vn",
        "url": "https://vanban.chinhphu.vn/documents/2025/06/202506/20250617/20250617143000/20250617143000.docx",
    },
    {
        "filename": "nghi-dinh-1632026ND-CP-huong-dan-luat-phong-chong-ma-tuy.docx",
        "title": "Nghị định 163/2026/NĐ-CP hướng dẫn thi hành Luật Phòng, chống ma túy",
        "source": "https://vanban.chinhphu.vn",
        "url": "https://vanban.chinhphu.vn/documents/2026/01/202601/20260115/20260115143000/20260115143000.docx",
    },
    {
        "filename": "nghi-dinh-28-2026-ND-CP-quy-dinh-cac-danh-muc-chat-ma-tuy-va-tien-chat.docx",
        "title": "Nghị định 28/2026/NĐ-CP quy định các danh mục chất ma túy và tiền chất",
        "source": "https://vanban.chinhphu.vn",
        "url": "https://vanban.chinhphu.vn/documents/2026/01/202601/20260108/20260108143000/20260108143000.docx",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str, timeout: int = 60) -> Path:
    """Tải file PDF/DOCX từ URL về DATA_DIR."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    filepath = DATA_DIR / filename
    filepath.write_bytes(response.content)
    print(f"✓ Đã tải: {filepath} ({filepath.stat().st_size:,} bytes)")
    return filepath


def verify_legal_files() -> list[Path]:
    """Kiểm tra các file pháp luật đã tồn tại và không rỗng."""
    setup_directory()
    valid_extensions = {".pdf", ".docx", ".doc"}
    files = [
        f for f in DATA_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]

    print(f"\n📁 Tổng số file pháp luật: {len(files)}")
    for f in sorted(files):
        size_kb = f.stat().st_size / 1024
        print(f"  • {f.name} ({size_kb:.1f} KB)")

    if len(files) < 3:
        print("\n⚠ Cần tối thiểu 3 file. Hãy tải thủ công từ các nguồn trong LEGAL_DOCUMENTS.")
    return files


def download_all(skip_existing: bool = True) -> list[Path]:
    """Tải toàn bộ văn bản trong LEGAL_DOCUMENTS về data/landing/legal/."""
    setup_directory()
    downloaded = []

    for doc in LEGAL_DOCUMENTS:
        filepath = DATA_DIR / doc["filename"]
        if skip_existing and filepath.exists() and filepath.stat().st_size > 1024:
            print(f"⊘ Đã có: {filepath.name}")
            downloaded.append(filepath)
            continue

        print(f"↓ Đang tải: {doc['title']}")
        try:
            downloaded.append(download_file(doc["url"], doc["filename"]))
        except Exception as e:
            print(f"  ✗ Lỗi tải {doc['filename']}: {e}")

    print(f"\n✓ Hoàn thành: {len(downloaded)}/{len(LEGAL_DOCUMENTS)} file")
    return downloaded


if __name__ == "__main__":
    download_all()
    verify_legal_files()
