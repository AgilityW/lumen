import os
import tempfile

import pytest

from lumen.exceptions import ParseError
from lumen.parsers.md import MDParser


class TestMDParser:
    @pytest.fixture
    def md_file(self):
        content = """# Chapter 1: Introduction

This is the first chapter.

## Section 1.1

Some details here.

# Chapter 2: Deep Dive

More technical content.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def empty_md_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("")
            path = f.name
        yield path
        os.unlink(path)

    def test_parse_valid_md(self, md_file):
        parser = MDParser()
        result = parser.parse(md_file)
        assert result.title
        assert result.book_format == "md"
        assert len(result.chapters) >= 2
        assert any("Introduction" in ch["title"] for ch in result.chapters)
        assert result.raw_text

    def test_parse_empty_raises(self, empty_md_file):
        parser = MDParser()
        with pytest.raises(ParseError, match="File is empty"):
            parser.parse(empty_md_file)

    def test_get_chapters(self, md_file):
        parser = MDParser()
        result = parser.parse(md_file)
        chapters = parser.get_chapters(result)
        assert len(chapters) >= 2

    def test_parse_nonexistent_file_raises(self):
        parser = MDParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path.md")

    def test_extract_sections_no_headings(self):
        parser = MDParser()
        sections = parser._extract_sections("Plain text without any markdown headings.")
        assert len(sections) == 1
        assert sections[0]["title"] == "Full Content"

    def test_extract_sections_multiple_headings(self):
        parser = MDParser()
        content = "# A\ncontent a\n## B\ncontent b\n# C\ncontent c"
        sections = parser._extract_sections(content)
        assert len(sections) >= 2


class TestMDParserEdgeCases:
    def test_non_utf8_file(self):
        parser = MDParser()
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"\xff\xfe\x00\x01")
            path = f.name
        try:
            with pytest.raises(ParseError, match="UTF-8"):
                parser.parse(path)
        finally:
            os.unlink(path)
