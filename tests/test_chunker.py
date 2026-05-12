import pytest

from lumen.core.chunker import (
    chunk_book,
    _chunk_book_default,
    _split_paragraphs,
    _split_by_section_markers,
    _split_by_qa,
    _chunk_podcast,
)


class TestChunker:
    def test_chunk_book_default_with_chapters(self):
        parsed = {
            "content_type": "book",
            "text": "",
            "chapters": [
                {"title": "Chapter 1", "text": "First chapter content. " * 100},
                {"title": "Chapter 2", "text": "Second chapter content. " * 100},
            ],
        }
        chunks = _chunk_book_default(parsed)
        assert len(chunks) >= 2
        assert all("source" in c for c in chunks)
        assert all("text" in c for c in chunks)

    def test_chunk_book_default_empty(self):
        parsed = {"content_type": "book", "text": "", "chapters": []}
        chunks = _chunk_book_default(parsed)
        assert chunks == []

    def test_split_paragraphs_simple(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = _split_paragraphs(text, "test")
        assert len(chunks) >= 1
        assert all(c["source"] == "test" for c in chunks)

    def test_split_paragraphs_empty(self):
        assert _split_paragraphs("", "test") == []

    def test_chunk_podcast_simple(self):
        text = "# Intro\nHello world.\n# Main\nCore content.\n# Outro\nBye."
        chunks = _chunk_podcast(text, target_chars=50000)
        assert len(chunks) >= 1
        assert all("source" in c and "text" in c for c in chunks)

    def test_chunk_podcast_empty(self):
        chunks = _chunk_podcast("", target_chars=50000)
        # empty string gets a default "Intro" section with empty text
        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0]["source"] == "Intro" and chunks[0]["text"] == "")


class TestSplitBySectionMarkers:
    def test_basic_split(self):
        text = "## Intro\nhello\n## Main\nworld"
        sections = _split_by_section_markers(text)
        assert len(sections) >= 2
        assert any("Intro" in title for title, _ in sections)
        assert any("Main" in title for title, _ in sections)

    def test_no_markers(self):
        text = "just plain text\nno headers here"
        sections = _split_by_section_markers(text)
        assert len(sections) == 1


class TestSplitByQA:
    def test_qa_boundaries(self):
        text = "Q: What is X?\nA: It's Y.\nQ: How about Z?\nA: Z is W."
        segments = _split_by_qa(text)
        assert len(segments) >= 1

    def test_no_qa_boundaries(self):
        text = "Just plain text without any Q&A markers."
        segments = _split_by_qa(text)
        assert segments == [text]


class TestChunkBook:
    def test_chunk_book_adds_ids(self):
        parsed = {
            "content_type": "book",
            "text": "book text. " * 500,
            "chapters": [
                {"title": "Ch1", "text": "content. " * 200},
            ],
        }
        result = chunk_book(parsed)
        assert "chunks" in result
        for idx, chunk in enumerate(result["chunks"]):
            assert chunk["id"] == f"chunk-{idx+1:04d}"
            assert chunk["index"] == idx
