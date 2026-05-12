"""Markdown parser — extracts text and structure from .md/.markdown files."""

from pathlib import Path
from typing import Any

from lumen.exceptions import ParseError
from lumen.parsers.base import BaseParser, ParsedBook


class MDParser(BaseParser):
    """Parse Markdown files. Validates content before processing."""

    def parse(self, path: str) -> dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            raise ParseError("Could not decode file as UTF-8.")

        if not content.strip():
            raise ParseError("File is empty.")

        chapters = self._extract_sections(content)

        return ParsedBook(
            title=Path(path).stem,
            book_format="md",
            chapters=chapters,
            metadata={},
            raw_text=content.strip(),
        )

    def get_chapters(self, parsed: ParsedBook) -> list[dict[str, Any]]:
        return parsed.chapters

    def _extract_sections(self, content: str) -> list[dict[str, Any]]:
        """Split Markdown into sections by heading boundaries."""
        import re

        sections: list[dict[str, Any]] = []
        lines = content.split("\n")
        current_section: dict[str, Any] | None = None
        current_body: list[str] = []

        heading_pattern = re.compile(r"^#{1,3}\s+(.+)$")

        for line in lines:
            m = heading_pattern.match(line)
            if m:
                # Save previous section
                if current_section is not None:
                    current_section["text"] = "\n".join(current_body).strip()
                    sections.append(current_section)

                current_section = {"title": m.group(1).strip(), "text": ""}
                current_body = []
            else:
                current_body.append(line)

        # Last section
        if current_section is not None:
            current_section["text"] = "\n".join(current_body).strip()
            sections.append(current_section)
        else:
            # No headings at all — treat whole file as one section
            sections.append({"title": "Full Content", "text": content.strip()})

        return sections
