"""Chunker — splits book/podcast text into segments.

For books: ~8K tokens per chunk (~32K chars), paragraph boundary splitting.
For podcasts: splits at Q&A boundaries, section markers, then paragraph boundaries.
"""

import re
from typing import Any

# Rough heuristic: 1 token ≈ 4 characters for English/Chinese mixed text
_CHARS_PER_TOKEN = 4
_TARGET_TOKENS = 8000
_TARGET_CHARS = _TARGET_TOKENS * _CHARS_PER_TOKEN  # ~32K
_MAX_CHARS = int(_TARGET_CHARS * 1.2)  # 20% tolerance


def chunk_book(parsed: dict[str, Any]) -> dict[str, Any]:
    """Chunk book text. Preserves chapter boundaries where possible.

    Returns the parsed dict with an added "chunks" list and content_type.
    """
    content_type = parsed.get("content_type", "unknown")
    text = parsed.get("text", "")

    if content_type == "podcast":
        # Use smaller target for podcasts so speaker-boundary splitting kicks in
        chunks = _chunk_podcast(text, target_chars=10000)
    elif content_type == "article":
        chunks = _chunk_book_default(parsed)
    elif content_type == "reference":
        chunks = _chunk_by_sections(parsed)
    else:
        # book / unknown — existing behavior
        chunks = _chunk_book_default(parsed)

    # Build chunk index
    for idx, chunk in enumerate(chunks):
        chunk["id"] = f"chunk-{idx+1:04d}"
        chunk["index"] = idx

    parsed["chunks"] = chunks
    return parsed


# ── Podcast chunking ────────────────────────────────────────────────────


# Q&A boundary: "**Q:" or "Q:" at start of line
_QA_BOUNDARY = re.compile(r"^\*{0,2}[Qq][:\uFF1A]", re.MULTILINE)
# Section markers like ## Summary, ## Takeaways, ## Transcript
_SECTION_MARKER = re.compile(r"^#{1,3}\s+\w+(?:\s+\w+)*$", re.MULTILINE)
# Timestamp line: [(00:00)] or [(00:00:00)]
_TIMESTAMP_LINE = re.compile(r"^\[\(\d{1,2}:\d{2}(?::\d{2})?\)\]")


def _chunk_podcast(text: str, target_chars: int = _TARGET_CHARS) -> list[dict[str, Any]]:
    """Chunk a podcast transcript at Q&A boundaries, section markers, then paragraph bounds."""
    # First: split into top-level sections (## headers)
    sections = _split_by_section_markers(text)

    chunks: list[dict[str, Any]] = []

    for section_title, section_text in sections:
        # Try splitting this section by Q&A boundaries
        qa_segments = _split_by_qa(section_text)

        # If we got multiple Q&A segments, use them; otherwise use the section as-is
        if len(qa_segments) > 1:
            segments = qa_segments
        else:
            segments = [section_text]

        for seg_text in segments:
            # If a single segment is too large, sub-chunk at speaker boundaries (preserves flow)
            max_chars_for_podcast = max(target_chars, min(len(seg_text) // 3, _MAX_CHARS))
            if len(seg_text) > max_chars_for_podcast:
                sub_chunks = _split_at_speaker_boundaries(seg_text, section_title, target_chars)
                chunks.extend(sub_chunks)
            else:
                chunks.append({"source": section_title, "text": seg_text.strip()})

    # Merge very small consecutive chunks (<2K chars) from the same source section
    # This combines short items like Highlights or Takeaways without undoing speaker splits.
    if chunks:
        merged = [chunks[0]]
        for c in chunks[1:]:
            if c["source"] == merged[-1]["source"] and len(merged[-1]["text"]) + len(c["text"]) < 2000:
                merged[-1]["text"] += "\n\n" + c["text"]
            else:
                merged.append(c)
        chunks = merged

    # Fallback: if no sections were found, treat as plain text
    if not chunks:
        chunks = _split_paragraphs(text, "full")

    return chunks


def _split_by_section_markers(text: str) -> list[tuple[str, str]]:
    """Split text at ## heading boundaries. Returns [(title, text), ...]."""
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_title = "Intro"
    current_lines: list[str] = []

    for line in lines:
        m = _SECTION_MARKER.match(line)
        if m:
            # Save previous section
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = m.group(0).lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return sections


def _split_by_qa(text: str) -> list[str]:
    """Split text at Q: or **Q: boundaries. Returns list of segment texts."""
    matches = list(_QA_BOUNDARY.finditer(text))
    if len(matches) <= 1:
        return [text]

    segments = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg = text[start:end].strip()
        if seg:
            segments.append(seg)

    return segments


# ── Generic paragraph split ────────────────────────────────────────────

def _split_paragraphs(text: str, source_label: str) -> list[dict[str, Any]]:
    """Split text at paragraph boundaries, targeting ~8K tokens per chunk."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[dict[str, Any]] = []
    current_chunk: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para) + 2  # +2 for newlines

        if current_len + para_len > _MAX_CHARS and current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({"source": source_label, "text": chunk_text})
            current_chunk = []
            current_len = 0

        if para_len > _MAX_CHARS:
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({"source": source_label, "text": chunk_text})
                current_chunk = []
                current_len = 0
            # Split oversized paragraph at sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sub_chunk: list[str] = []
            sub_len = 0
            for sent in sentences:
                sent_len = len(sent) + 1
                if sub_len + sent_len > _MAX_CHARS and sub_chunk:
                    chunks.append({"source": source_label, "text": " ".join(sub_chunk)})
                    sub_chunk = []
                    sub_len = 0
                sub_chunk.append(sent)
                sub_len += sent_len
            if sub_chunk:
                chunks.append({"source": source_label, "text": " ".join(sub_chunk)})
        else:
            current_chunk.append(para)
            current_len += para_len

    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunks.append({"source": source_label, "text": chunk_text})

    return chunks


# ── Legacy book chunking ───────────────────────────────────────────────



def _split_at_speaker_boundaries(text: str, source_label: str, max_chars: int) -> list[dict[str, Any]]:
    """Split a large podcast segment at speaker boundaries (**Speaker:** lines).
    
    Each speaker turn is preserved as a unit. Adjacent short turns are merged
    up to max_chars. Falls back to paragraph split if no speaker boundaries found.
    """
    speaker_pattern = re.compile(r'^\*\*[^*]+:\*\*', re.MULTILINE)
    matches = list(speaker_pattern.finditer(text))
    
    # No speaker boundaries or very few — fall back to paragraph split
    if len(matches) < 2:
        return _split_paragraphs(text, source_label)
    
    # Extract speaker turns
    turns = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        turn_text = text[start:end].strip()
        if turn_text:
            turns.append(turn_text)
    
    # Merge turns into chunks
    chunks = []
    current_chunk = []
    current_len = 0
    
    for turn in turns:
        turn_len = len(turn)
        if current_len + turn_len > max_chars and current_chunk:
            chunks.append({"source": source_label, "text": "\n\n".join(current_chunk)})
            current_chunk = []
            current_len = 0
        
        # Single turn exceeding max — force split at paragraph boundaries within it
        if turn_len > max_chars:
            if current_chunk:
                chunks.append({"source": source_label, "text": "\n\n".join(current_chunk)})
                current_chunk = []
                current_len = 0
            # Split this turn at sentence boundaries
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", turn) if s.strip()]
            sub: list[str] = []
            sub_len = 0
            for sent in sentences:
                sent_len = len(sent) + 1
                if sub_len + sent_len > max_chars and sub:
                    chunks.append({"source": source_label, "text": " ".join(sub)})
                    sub = []
                    sub_len = 0
                sub.append(sent)
                sub_len += sent_len
            if sub:
                chunks.append({"source": source_label, "text": " ".join(sub)})
        else:
            current_chunk.append(turn)
            current_len += turn_len
    
    if current_chunk:
        chunks.append({"source": source_label, "text": "\n\n".join(current_chunk)})
    
    return chunks

def _chunk_book_default(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Default chunking — per-chapter with paragraph boundary splitting."""
    chapters = parsed.get("chapters", [])
    chunks: list[dict[str, Any]] = []

    if chapters:
        for i, chapter in enumerate(chapters):
            text = chapter.get("text", "") or ""
            chapter_chunks = _split_paragraphs(text, chapter.get("title", f"ch-{i+1}"))
            chunks.extend(chapter_chunks)
    else:
        chunks = _split_paragraphs(parsed.get("text", ""), "full")

    return chunks


def _chunk_by_sections(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Reference chunking — each section is a chunk, no sub-splitting."""
    chapters = parsed.get("chapters", [])
    chunks: list[dict[str, Any]] = []

    if chapters:
        for chapter in chapters:
            text = chapter.get("text", "") or ""
            chunks.append({
                "source": chapter.get("title", "section"),
                "text": text,
            })
    else:
        chunks.append({"source": "full", "text": parsed.get("text", "")})

    return chunks
