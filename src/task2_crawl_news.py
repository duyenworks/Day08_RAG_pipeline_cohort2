"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Bài báo về nghệ sĩ Việt Nam liên quan tới ma tuý (VnExpress, Ngôi sao)
ARTICLE_URLS = [
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
    "https://vnexpress.net/hoai-dj-linh-an-tu-hinh-5071068.html",
    "https://ngoisao.vnexpress.net/ca-si-long-nhat-va-son-ngoc-minh-bi-bat-5076069.html",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-va-ca-si-chi-dan-bi-bat-4814295.html",
]


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(text: str, max_len: int = 60) -> str:
    """Chuyển tiêu đề thành tên file an toàn."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:max_len] or "article"


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)

        title = "Unknown"
        if result.metadata:
            title = result.metadata.get("title") or result.metadata.get("og:title") or title

        content = result.markdown or result.cleaned_html or ""
        if not content.strip():
            raise ValueError(f"Không lấy được nội dung từ: {url}")

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now(timezone.utc).isoformat(),
            "content_markdown": content,
        }


def save_article(article: dict, index: int) -> Path:
    """Lưu bài báo thành file JSON."""
    slug = _slugify(article["title"])
    filename = f"article_{index:02d}_{slug}.json"
    filepath = DATA_DIR / filename
    filepath.write_text(
        json.dumps(article, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return filepath


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()
    saved = []

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
            filepath = save_article(article, i)
            saved.append(filepath)
            print(f"  ✓ Saved: {filepath.name} — {article['title'][:60]}")
        except Exception as e:
            print(f"  ✗ Lỗi: {e}")

    print(f"\n✓ Hoàn thành: {len(saved)}/{len(ARTICLE_URLS)} bài báo")
    return saved


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
