"""Base Analyzer interface — all LLM backends implement this."""

from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass


@dataclass
class ArchetypeDef:
    description: str
    check_prompt: str
    importance: str = "medium"
    expected_topics: list[str] | None = None


class BaseAnalyzer(ABC):
    """Abstract interface for LLM backends. All 7 methods must be implemented.

    Content-type aware: adapts extraction strategy based on content_type param.
    """

    @abstractmethod
    def skeletonize(self, chunks: list, framework: dict, content_type: str = "unknown") -> list[dict]:
        ...

    @abstractmethod
    def analyze_chapter(self, chunk: str, context: dict) -> dict[str, Any]:
        ...

    @abstractmethod
    def synthesize(self, analyses: list[dict]) -> dict[str, Any]:
        ...

    @abstractmethod
    def gap_analyze(
        self,
        skeleton: list[dict],
        archetypes: dict,
        raw_content: str,
        content_type: str = "unknown",
    ) -> dict:
        ...

    @abstractmethod
    def refine_skeleton(
        self,
        initial_skeleton: list[dict],
        new_topics: list[dict],
        content_type: str = "unknown",
    ) -> list[dict]:
        ...

    @abstractmethod
    def check_coverage(self, skeleton: list[dict], archetypes: dict) -> list[dict]:
        ...

    @abstractmethod
    def adapt_archetypes(
        self,
        skeleton: list[dict],
        raw_content: str = "",
        content_type: str = "unknown",
    ) -> dict:
        ...
