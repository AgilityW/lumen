"""Tests for renderers — ObsidianRenderer, MindmapRenderer, WeChatRenderer."""

import os
import tempfile

import pytest

from lumen.renderers.mindmap import MindmapRenderer
from lumen.renderers.obsidian import ObsidianRenderer
from lumen.renderers.html import WeChatRenderer


# ── Shared fixture ─────────────────────────────────────────────────────

_SAMPLE_SYNTHESIS = {
    "book_summary": "A book about testing software.",
    "core_concepts": [
        {"name": "Unit Testing", "definition": "Testing individual components.", "importance": "high"},
        {"name": "Integration Testing", "definition": "Testing component interactions.", "importance": "high"},
    ],
    "key_arguments": [
        "Tests should be fast and focused.",
        "Mock external dependencies.",
    ],
    "relationship_map": [
        {"from": "Unit Testing", "to": "Integration Testing", "type": "builds-upon"},
    ],
    "reading_notes": [
        "Chapter 1 covers the basics.",
        {"note": "Focus on test doubles in chapter 3."},
    ],
}


# ── MindmapRenderer ────────────────────────────────────────────────────


class TestMindmapRenderer:
    def test_render_basic(self):
        renderer = MindmapRenderer()
        result = renderer.render(_SAMPLE_SYNTHESIS)
        assert result.startswith("flowchart LR")
        assert "B((Book))" in result
        assert "Unit Testing" in result
        assert "Integration Testing" in result
        assert "subgraph Concepts" in result
        assert "subgraph Arguments" in result
        assert "subgraph Relationships" in result

    def test_render_empty_synthesis(self):
        renderer = MindmapRenderer()
        result = renderer.render({})
        assert result == "flowchart LR\n  B((Book))"

    def test_render_concept_limit_15(self):
        many_concepts = [{"name": f"Concept {i}"} for i in range(20)]
        result = MindmapRenderer().render({"core_concepts": many_concepts})
        assert "Concept 0" in result
        assert "Concept 14" in result
        assert "Concept 15" not in result

    def test_render_argument_limit_10(self):
        many_args = [f"Argument {i}" for i in range(15)]
        result = MindmapRenderer().render({"key_arguments": many_args})
        assert "Argument 0" in result
        assert "Argument 9" in result
        assert "Argument 10" not in result

    def test_render_relationship_limit_10(self):
        many_rels = [{"from": f"A{i}", "to": f"B{i}", "type": "depends"} for i in range(15)]
        result = MindmapRenderer().render({"relationship_map": many_rels,
                                            "core_concepts": [], "key_arguments": []})
        # Only count arrow lines (-- ... -->), not subgraph headers like "B --> Relationships"
        arrow_lines = [l for l in result.split("\n") if "-- depends -->" in l]
        assert len(arrow_lines) <= 10

    def test_render_dedup_relationships(self):
        rels = [
            {"from": "A", "to": "B", "type": "depends"},
            {"from": "A", "to": "B", "type": "depends"},
        ]
        result = MindmapRenderer().render({"relationship_map": rels,
                                            "core_concepts": [], "key_arguments": []})
        arrow_lines = [l for l in result.split("\n") if "-- depends -->" in l]
        assert len(arrow_lines) == 1

    def test_render_special_chars_in_name(self):
        synthesis = {
            "core_concepts": [{"name": 'Concept with "quotes" and /slashes/'}],
        }
        result = MindmapRenderer().render(synthesis)
        assert "quotes" in result
        assert "Concept" in result


# ── ObsidianRenderer ───────────────────────────────────────────────────


class TestObsidianRenderer:
    @pytest.fixture
    def vault_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp

    def test_render_creates_main_note(self, vault_dir):
        renderer = ObsidianRenderer(vault_path=vault_dir)
        path = renderer.render("test-book", _SAMPLE_SYNTHESIS)
        assert os.path.exists(path)
        content = open(path).read()
        assert "# test-book" in content
        assert "## Summary" in content
        assert "A book about testing software." in content
        assert "## Core Concepts" in content
        assert "[[unit-testing]]" in content
        assert "[[integration-testing]]" in content
        assert "## Key Arguments" in content
        assert "## Relationship Map" in content
        assert "## Reading Notes" in content

    def test_render_creates_concept_notes(self, vault_dir):
        renderer = ObsidianRenderer(vault_path=vault_dir)
        renderer.render("test-book", _SAMPLE_SYNTHESIS)
        book_dir = os.path.join(vault_dir, "Books", "test-book")
        assert os.path.exists(os.path.join(book_dir, "unit-testing.md"))
        assert os.path.exists(os.path.join(book_dir, "integration-testing.md"))

    def test_concept_note_content(self, vault_dir):
        renderer = ObsidianRenderer(vault_path=vault_dir)
        renderer.render("test-book", _SAMPLE_SYNTHESIS)
        note = open(os.path.join(vault_dir, "Books", "test-book", "unit-testing.md")).read()
        assert 'title: "Unit Testing"' in note
        assert "tags: [concept, book/test-book]" in note
        assert "Testing individual components." in note
        assert "Sourced from: **[[test-book]]**" in note

    def test_render_empty_synthesis(self, vault_dir):
        renderer = ObsidianRenderer(vault_path=vault_dir)
        path = renderer.render("empty-book", {})
        content = open(path).read()
        assert "# empty-book" in content
        assert "No summary available." in content

    def test_write_mindmap_appends(self, vault_dir):
        renderer = ObsidianRenderer(vault_path=vault_dir)
        renderer.render("test-book", _SAMPLE_SYNTHESIS)
        renderer.write_mindmap("test-book", "flowchart LR\n  B((Book))")
        content = open(os.path.join(vault_dir, "Books", "test-book", "test-book.md")).read()
        assert "## Mind Map" in content
        assert "flowchart LR" in content

    def test_write_mindmap_replaces(self, vault_dir):
        renderer = ObsidianRenderer(vault_path=vault_dir)
        renderer.render("test-book", _SAMPLE_SYNTHESIS)
        renderer.write_mindmap("test-book", "flowchart LR\n  B((Book))")
        renderer.write_mindmap("test-book", "flowchart LR\n  B((Updated))")
        content = open(os.path.join(vault_dir, "Books", "test-book", "test-book.md")).read()
        assert "Updated" in content
        assert content.count("## Mind Map") == 1

    def test_write_mindmap_no_note_does_nothing(self, vault_dir):
        renderer = ObsidianRenderer(vault_path=vault_dir)
        renderer.write_mindmap("nonexistent", "flowchart LR")

    def test_concept_dedup_by_slug(self, vault_dir):
        synthesis = {
            "core_concepts": [
                {"name": "Machine Learning"},
                {"name": "machine-learning"},
            ],
        }
        renderer = ObsidianRenderer(vault_path=vault_dir)
        renderer.render("dedup-test", synthesis)
        book_dir = os.path.join(vault_dir, "Books", "dedup-test")
        files = [f for f in os.listdir(book_dir) if f.endswith(".md")]
        assert len(files) == 2

    def test_render_no_concepts(self, vault_dir):
        synthesis = {"book_summary": "Just a summary."}
        renderer = ObsidianRenderer(vault_path=vault_dir)
        path = renderer.render("no-concepts", synthesis)
        content = open(path).read()
        assert "## Core Concepts" not in content


# ── WeChatRenderer ─────────────────────────────────────────────────────


class TestWeChatRenderer:
    def test_render_contains_title(self):
        renderer = WeChatRenderer(book_title="Test Book")
        html = renderer.render(_SAMPLE_SYNTHESIS)
        # WeChatRenderer wraps in <section>, not a full HTML document
        assert "<section" in html
        assert "Test Book" in html

    def test_render_sections(self):
        renderer = WeChatRenderer()
        html = renderer.render(_SAMPLE_SYNTHESIS)
        assert "核心概念" in html
        assert "关键论点" in html
        assert "关系图谱" in html
        assert "阅读笔记" in html

    def test_render_summary(self):
        renderer = WeChatRenderer()
        html = renderer.render(_SAMPLE_SYNTHESIS)
        assert "A book about testing software." in html

    def test_render_concepts(self):
        renderer = WeChatRenderer()
        html = renderer.render(_SAMPLE_SYNTHESIS)
        assert "Unit Testing" in html
        assert "Integration Testing" in html
        assert "Testing individual components." in html

    def test_render_empty_synthesis(self):
        renderer = WeChatRenderer()
        html = renderer.render({})
        assert "<section" in html

    def test_render_no_inline_js(self):
        renderer = WeChatRenderer()
        html = renderer.render(_SAMPLE_SYNTHESIS)
        assert "<script" not in html

    def test_render_all_inline_css(self):
        renderer = WeChatRenderer()
        html = renderer.render(_SAMPLE_SYNTHESIS)
        assert "<style>" not in html
        assert '<link rel="stylesheet"' not in html
        assert 'style="' in html

    def test_render_no_markdown_leftover(self):
        renderer = WeChatRenderer()
        html = renderer.render(_SAMPLE_SYNTHESIS)
        assert "[[" not in html
        assert "```" not in html
        assert "**" not in html
