"""PDF parser — extracts text and structure from text-layer PDFs."""

import re
from pathlib import Path
from typing import Any

from lumen.exceptions import ParseError
from lumen.parsers.base import BaseParser, ParsedBook

_CHAPTER_PATTERN = re.compile(
    r'^(chapter|ch\.|section|part|module)\s+(\d+|[ivxlcdm]+)\s*[.:]?\s*(.*)$',
    re.IGNORECASE,
)


class PDFParser(BaseParser):
    """Parse text-based PDFs. Exits with guidance if no text layer found."""

    def parse(self, path: str) -> dict[str, Any]:
        import fitz  # pymupdf

        doc = fitz.open(path)
        metadata = doc.metadata or {}
        pages: list[dict[str, Any]] = []

        total_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if text:
                total_text += text + "\n\n"
            pages.append({
                "page": page_num + 1,
                "text": text,
                "has_text": bool(text),
            })
        doc.close()

        if not total_text.strip():
            raise ParseError("No text layer found. Use a text-based PDF or run OCR first.")

        chapters = self._infer_chapters(pages)

        return ParsedBook(
            title=metadata.get("title", Path(path).stem),
            book_format="pdf",
            chapters=chapters,
            metadata={"author": metadata.get("author", ""), "pages": len(pages)},
            raw_text=total_text.strip(),
        )

    def get_chapters(self, parsed: ParsedBook) -> list[dict[str, Any]]:
        return parsed.chapters

    def _infer_chapters(self, pages: list[dict]) -> list[dict[str, Any]]:
        """Heuristic: detect likely chapter/section headings."""
        chapters: list[dict[str, Any]] = []
        seen = set()

        for page in pages:
            text = page.get("text", "")
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                m = _CHAPTER_PATTERN.match(line)
                if m and line not in seen:
                    seen.add(line)
                    chapters.append({
                        "title": line,
                        "page": page["page"],
                    })
        return chapters
