"""Parser dispatch — routes books to the correct format parser."""

import os
from pathlib import Path
from typing import Any

from lumen.exceptions import ParseError
from lumen.parsers.epub import EPUBParser
from lumen.parsers.md import MDParser
from lumen.parsers.pdf import PDFParser

SUPPORTED_EXTENSIONS = {
    ".pdf": PDFParser,
    ".epub": EPUBParser,
    ".md": MDParser,
    ".markdown": MDParser,
}


def parse_book(path: str) -> dict[str, Any]:
    """Detect format, parse, and run basic validation.

    Returns the full parsed dict with text, chapters, and metadata.
    Exits with guidance on validation failures.
    """
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise ParseError(f"File not found: {path}")

    ext = Path(path).suffix.lower()
    parser_cls = SUPPORTED_EXTENSIONS.get(ext)

    if parser_cls is None:
        raise ParseError(
            f"Unsupported format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    parser = parser_cls()  # type: ignore[abstract]
    pb = parser.parse(path)
    # Convert ParsedBook dataclass -> dict (map to downstream expected keys)
    parsed = {
        "format": pb.book_format,
        "title": pb.title,
        "author": pb.metadata.get("author", ""),
        "pages": pb.metadata.get("pages", 0),
        "chapters": pb.chapters,
        "text": pb.raw_text,
    }
    parsed["source_path"] = path
    return parsed


def build_structure(parsed: dict[str, Any]) -> dict[str, Any]:
    """Build structure.json from parsed book data.

    The structure is a chapter/section map used during skeleton extraction
    to efficiently locate content.
    """
    chapters = parsed.get("chapters", [])
    chunks = parsed.get("chunks", [])

    return {
        "title": parsed.get("title", ""),
        "format": parsed.get("format", ""),
        "author": parsed.get("author", ""),
        "total_chapters": len(chapters),
        "total_chunks": len(chunks),
        "chapters": [
            {"id": f"ch-{i+1:02d}", "title": ch.get("title", f"Chapter {i+1}")}
            for i, ch in enumerate(chapters)
        ],
        "chunk_count_by_source": _count_by_source(chunks),
    }


def _count_by_source(chunks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in chunks:
        src = c.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts
