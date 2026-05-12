"""Base parser interface — all format parsers implement this."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedBook:
    """Standard output from all parsers."""
    title: str
    book_format: str  # "pdf" | "epub" | "md"
    chapters: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""


class BaseParser(ABC):
    @abstractmethod
    def parse(self, path: str) -> ParsedBook:
        ...

    def get_chapters(self, parsed: ParsedBook) -> list[dict[str, Any]]:
        return parsed.chapters
