"""EPUB parser — extracts text and structure from EPUB books."""

from pathlib import Path
from typing import Any

from lumen.exceptions import ParseError
from lumen.parsers.base import BaseParser, ParsedBook


class EPUBParser(BaseParser):
    """Parse EPUB files. Exits with guidance on corruption."""

    def parse(self, path: str) -> ParsedBook:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        try:
            book = epub.read_epub(path)
        except Exception as exc:
            raise ParseError(f"Could not unpack EPUB. File may be corrupted: {exc}")

        title = book.get_metadata("DC", "title")
        title = title[0][0] if title else Path(path).stem

        author = book.get_metadata("DC", "creator")
        author = author[0][0] if author else ""

        chapters: list[dict[str, Any]] = []
        total_text = ""
        toc_items = list(book.toc)
        # Preprocess TOC into {filename: title} dict for O(1) lookup
        toc_map: dict[str, str] = {}
        for toc_entry in toc_items:
            if isinstance(toc_entry, tuple) and len(toc_entry) >= 2:
                href = toc_entry[1].get("href", "") if hasattr(toc_entry[1], "get") else ""
                href_name = Path(href).name
                if href_name:
                    toc_map[href_name] = toc_entry[0]

        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n").strip()
            if not text:
                continue

            # Use TOC entry as chapter title if available (O(1) dict lookup)
            title_text = Path(item.get_name()).stem.replace("-", " ").title()
            item_name = Path(item.get_name()).name
            toc_title = toc_map.get(item_name)
            if toc_title:
                title_text = toc_title

            chapters.append({
                "title": title_text,
                "text": text,
            })
            total_text += text + "\n\n"

        if not total_text.strip():
            print("[WARN] EPUB appears to contain no readable text.")

        return ParsedBook(
            title=title,
            book_format="epub",
            chapters=chapters,
            metadata={"author": author},
            raw_text=total_text.strip(),
        )

    def get_chapters(self, parsed: ParsedBook) -> list[dict[str, Any]]:
        return parsed.chapters
