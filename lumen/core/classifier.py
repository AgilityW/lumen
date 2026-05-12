"""Content type classifier — detects podcast/book/article/reference from parsed text.

Phase 0: runs after parsing, before chunking. Uses heuristic signal detection.
"""

import re
from typing import Any

# ── Signal patterns ────────────────────────────────────────────────────

PODCAST_SIGNALS = {
    # Timecode patterns like (00:00), [00:00:00], (00:12:34)
    "timecodes": re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    # Q&A structure markers
    "qa_markers": re.compile(r"^[Qq][:\uFF1A]|^[Aa][:\uFF1A]", re.MULTILINE),
    # Speaker labels
    "speaker_labels": re.compile(
        r"^(Speaker\s+\d+[:\uFF1A])"
        r"|^(Host[:\uFF1A])|^(Guest[:\uFF1A])"
        r"|^(\*\*[^*]+\*\*:)"
        r"|^([A-Z][a-z]+\s+[A-Z][a-z]+[:\uFF1A])",
        re.MULTILINE,
    ),
    # Podcast metadata
    "podcast_meta": re.compile(
        r"podcast[\s\-:]*", re.IGNORECASE
    ),
    # Section markers typical of podcast notes
    "takeaways_section": re.compile(r"^#+\s*(Takeaway|Highlight|Key Insight)", re.MULTILINE | re.IGNORECASE),
    "transcript_section": re.compile(r"^#+\s*(Transcript|Full\s+Transcript)", re.MULTILINE | re.IGNORECASE),
    "qa_section": re.compile(r"^#+\s*(Q\s*&\s*A|Q and A|Questions?|Answers?)", re.MULTILINE | re.IGNORECASE),
}

BOOK_SIGNALS = {
    "chapter_marker": re.compile(r"^(Chapter|Part|Section)\s+\d", re.MULTILINE | re.IGNORECASE),
    "toc_marker": re.compile(r"^#+\s*(Table of Contents|Contents|目[录錄])", re.MULTILINE | re.IGNORECASE),
    "isbn_marker": re.compile(r"\bISBN[\s:-]*\d[\d\s-]{9,}\b", re.IGNORECASE),
    "publisher_marker": re.compile(r"^#+\s*(Publisher|Published by|Copyright)", re.MULTILINE | re.IGNORECASE),
    "chapter_preface": re.compile(r"^#+\s*(Introduction|Preface|Foreword|Acknowledgments)", re.MULTILINE | re.IGNORECASE),
}

ARTICLE_SIGNALS = {
    "abstract_marker": re.compile(r"^#+\s*(Abstract|Summary|TL;DR)", re.MULTILINE | re.IGNORECASE),
    "references_marker": re.compile(r"^#+\s*(References|Bibliography|Citations)", re.MULTILINE | re.IGNORECASE),
    "introduction_marker": re.compile(r"^#+\s*(Introduction|Background|Motivation)", re.MULTILINE | re.IGNORECASE),
    "conclusion_marker": re.compile(r"^#+\s*(Conclusion|Discussion|Future Work)", re.MULTILINE | re.IGNORECASE),
}


def classify(text: str) -> str:
    """Detect content type from parsed text.

    Returns one of: "podcast", "book", "article", "reference", "unknown"
    """
    scores = {
        "podcast": _score_signals(text, PODCAST_SIGNALS),
        "book": _score_signals(text, BOOK_SIGNALS),
        "article": _score_signals(text, ARTICLE_SIGNALS),
    }

    # Weight adjustments
    lines = text.split("\n")
    # Podcast: high density of timecodes is very strong signal
    timecode_matches = list(PODCAST_SIGNALS["timecodes"].finditer(text))
    if timecode_matches:
        # If more than 3 timecodes per 100 lines, very likely podcast
        density = len(timecode_matches) / max(1, len(lines)) * 100
        if density > 3:
            scores["podcast"] += 15

    # Podcast: if Q&A section and speaker labels both present, very strong
    if scores["podcast"] > 0 and PODCAST_SIGNALS["speaker_labels"].search(text):
        scores["podcast"] += 10

    # Book: chapter markers are very strong signal
    chapter_count = len(BOOK_SIGNALS["chapter_marker"].findall(text))
    if chapter_count >= 3:
        scores["book"] += chapter_count * 3

    best = max(scores, key=scores.get)
    best_score = scores[best]

    if best_score == 0:
        return "unknown"

    # If runner-up is within 5 points, it's ambiguous — go with "unknown"
    runner_up = sorted(scores.values(), reverse=True)
    if len(runner_up) > 1 and best_score - runner_up[1] < 5:
        return "unknown"

    return best


def _score_signals(text: str, signal_group: dict) -> int:
    """Count matches for a group of signals. Returns weighted score."""
    score = 0
    for name, pattern in signal_group.items():
        matches = pattern.findall(text)
        if matches:
            score += len(matches)
    return score


def get_framework_name(content_type: str) -> str:
    """Map content type to framework file name."""
    mapping = {
        "podcast": "podcast",
        "book": "book",
        "article": "article",
        "reference": "reference",
        "unknown": "default",
    }
    return mapping.get(content_type, "default")
