"""Tests for analyzer core functions — mock LLM backend, test quality checks, GATE review."""

import json
import os
import tempfile
import unittest
import pytest
from unittest.mock import patch, MagicMock

from lumen.core.analyzer import (
    run_skeletonize,
    run_deep_read,
    run_synthesis,
    _check_skeleton_quality,
    _fill_missing_fields,
    _ensure_skeleton_field_safety,
    _build_full_text,
    present_skeleton_for_review,
    _present_book_skeleton,
    _present_podcast_skeleton,
)
from lumen.core.state import CheckpointManager


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_analyzer():
    mock = MagicMock()
    mock.skeletonize.return_value = [
        {"topic": "Topic 1", "claims": ["Claim 1"], "relationships": ["rel"], "prerequisites": []},
        {"topic": "Topic 2", "claims": ["Claim 2"], "relationships": ["rel"], "prerequisites": []},
    ]
    mock.analyze_chapter.return_value = {
        "chapter_id": "ch-01",
        "topics_covered": [{"topic": "T1", "key_insights": ["insight"]}],
    }
    mock.synthesize.return_value = {
        "book_summary": "Test summary.",
        "core_concepts": [{"name": "Concept 1", "definition": "Def 1"}],
        "key_arguments": ["Arg 1"],
        "relationship_map": [],
    }
    return mock


@pytest.fixture
def sample_chunks():
    return [
        {"source": "ch1", "text": "Content of chapter one. " * 100},
        {"source": "ch2", "text": "Content of chapter two. " * 100},
    ]


@pytest.fixture
def sample_skeleton():
    return [
        {"topic": "T1", "claims": ["c1"], "relationships": [], "prerequisites": []},
        {"topic": "T2", "claims": ["c2"], "relationships": [], "prerequisites": []},
    ]


@pytest.fixture
def work_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


# ── _build_full_text ───────────────────────────────────────────────────

class TestBuildFullText:
    def test_basic_concat(self):
        chunks = [{"source": "ch1", "text": "Hello"}, {"source": "ch2", "text": "World"}]
        result = _build_full_text(chunks)
        assert "=== ch1 ===" in result
        assert "Hello" in result
        assert "=== ch2 ===" in result
        assert "World" in result

    def test_truncated(self):
        chunks = [{"source": "ch1", "text": "Hello World"}]
        result = _build_full_text(chunks, max_chars=5)
        assert len(result) <= 5

    def test_empty_chunks(self):
        assert _build_full_text([]) == ""


# ── _check_skeleton_quality ────────────────────────────────────────────

class TestCheckSkeletonQuality:
    def test_good_skeleton(self):
        skeleton = [
            {"topic": "T1", "claims": ["c1"], "relationships": ["r1"], "prerequisites": ["p1"]},
            {"topic": "T2", "claims": ["c2"], "relationships": ["r2"], "prerequisites": []},
        ]
        ok, issues = _check_skeleton_quality(skeleton, {
            "min_topics": 2, "require_claims": True, "require_relationships": True,
        })
        assert ok is True
        assert issues == []

    def test_too_few_topics(self):
        skeleton = [{"topic": "T1", "claims": ["c1"]}]
        ok, issues = _check_skeleton_quality(skeleton, {"min_topics": 3})
        assert ok is False
        assert "need at least" in issues[0]

    def test_missing_claims_under_threshold(self):
        skeleton = [{"topic": "T1", "claims": []}]
        ok, issues = _check_skeleton_quality(skeleton, {"require_claims": True})
        assert ok is False

    def test_missing_relationships(self):
        skeleton = [{"topic": "T1", "claims": ["c1"], "relationships": []}]
        ok, issues = _check_skeleton_quality(skeleton, {"require_relationships": True})
        assert ok is False
        assert any("relationship" in i.lower() for i in issues)

    def test_soft_max_topics_warning(self):
        skeleton = [{"topic": f"T{i}"} for i in range(15)]
        ok, issues = _check_skeleton_quality(skeleton, {"soft_max_topics": 10})
        assert ok is True
        assert "soft" in issues[0].lower()

    def test_hard_max_topics_exceeded(self):
        skeleton = [{"topic": f"T{i}"} for i in range(15)]
        ok, issues = _check_skeleton_quality(skeleton, {"max_topics": 10})
        assert ok is False
        assert "absolute max" in issues[0]

    def test_empty_rules(self):
        ok, issues = _check_skeleton_quality([], {})
        assert ok is True
        assert issues == []

    def test_require_core_argument_podcast(self):
        skeleton = [{"topic": "T1"}, {"topic": "T2", "core_argument": "arg"}]
        ok, issues = _check_skeleton_quality(skeleton, {"require_core_argument": True}, content_type="podcast")
        assert ok is False
        assert "T1" in issues[0]

    def test_require_timestamp_podcast_under_threshold(self):
        # 0 out of 2 with timestamp → 0 < 1 (50% of 2) → fail
        skeleton = [{"topic": "T1"}, {"topic": "T2"}]
        ok, issues = _check_skeleton_quality(skeleton, {"require_timestamp": True}, content_type="podcast")
        assert ok is False

    def test_require_timestamp_podcast_above_threshold(self):
        skeleton = [{"topic": "T1", "timestamp": "00:05"}]
        ok, issues = _check_skeleton_quality(skeleton, {"require_timestamp": True}, content_type="podcast")
        assert ok is True


# ── _fill_missing_fields ───────────────────────────────────────────────

class TestFillMissingFields:
    def test_fill_podcast_fields(self):
        topic = {"topic": "Test"}
        result = _fill_missing_fields(topic, content_type="podcast")
        assert result.get("core_argument") == "No explicit core argument extracted."
        assert result.get("evidence_chain") == []
        assert result.get("tension_point") == "No tension point explicitly identified."
        assert result.get("key_quotes") == []
        assert result.get("timestamp") == ""
        assert result.get("relationships") == []

    def test_fill_book_fields(self):
        topic = {"topic": "Test"}
        result = _fill_missing_fields(topic, content_type="book")
        assert result.get("claims") == []
        assert result.get("relationships") == []
        assert result.get("prerequisites") == []

    def test_preserve_existing_fields(self):
        topic = {"topic": "Test", "claims": ["existing"], "prerequisites": ["pre"]}
        result = _fill_missing_fields(topic, content_type="book")
        assert result["claims"] == ["existing"]
        assert result["prerequisites"] == ["pre"]

    def test_unknown_type(self):
        topic = {"topic": "Test"}
        result = _fill_missing_fields(topic, content_type="unknown")
        assert result.get("claims") == []


# ── _ensure_skeleton_field_safety ──────────────────────────────────────

class TestEnsureSkeletonFieldSafety:
    def test_none_values_replaced(self):
        skeleton = [{"topic": "T1", "claims": None}]
        result = _ensure_skeleton_field_safety(skeleton)
        assert result[0]["claims"] == []

    def test_missing_keys_added(self):
        skeleton = [{"topic": "T1"}]
        result = _ensure_skeleton_field_safety(skeleton)
        assert "claims" in result[0]
        assert "relationships" in result[0]

    def test_existing_values_preserved(self):
        skeleton = [{"topic": "T1", "claims": ["good"]}]
        result = _ensure_skeleton_field_safety(skeleton)
        assert result[0]["claims"] == ["good"]

    def test_noop_on_good_skeleton(self):
        skeleton = [{"topic": "T1", "claims": [], "relationships": [], "prerequisites": []}]
        result = _ensure_skeleton_field_safety(skeleton)
        assert result == skeleton


# ── present_skeleton_for_review ───────────────────────────────────────

class TestPresentSkeletonForReview:
    def test_auto_approve(self):
        with patch.dict(os.environ, {"LUMEN_AUTO_APPROVE": "1"}):
            decision, feedback = present_skeleton_for_review(
                skeleton=[{"topic": "T1"}],
                book_title="Test Book",
            )
            assert decision == "approve"
            assert feedback == ""

    def test_auto_approve_true(self):
        with patch.dict(os.environ, {"LUMEN_AUTO_APPROVE": "true"}):
            decision, _ = present_skeleton_for_review(
                skeleton=[{"topic": "T1"}],
                book_title="Test",
            )
            assert decision == "approve"

    def test_auto_approve_not_set_defaults_to_interactive(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _="": "a")
        with patch.dict(os.environ, {}, clear=True):
            decision, _ = present_skeleton_for_review(
                skeleton=[{"topic": "T1"}],
                book_title="Test",
            )
            assert decision == "approve"

    def test_interactive_quit(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _="": "q")
        with patch.dict(os.environ, {}, clear=True):
            decision, _ = present_skeleton_for_review(
                skeleton=[{"topic": "T1"}],
                book_title="Test",
            )
            assert decision == "quit"

    def test_interactive_redo(self, monkeypatch):
        inputs = iter(["r", "add more depth"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        with patch.dict(os.environ, {}, clear=True):
            decision, feedback = present_skeleton_for_review(
                skeleton=[{"topic": "T1"}],
                book_title="Test",
            )
            assert decision == "redo"
            assert feedback == "add more depth"


# ── _present_book_skeleton / _present_podcast_skeleton ────────────────

class TestPresentBookSkeleton:
    def test_output_contains_topics(self, capsys):
        skeleton = [{"topic": "Test", "claims": ["c1"], "relationships": ["r1"], "prerequisites": ["p1"]}]
        _present_book_skeleton(skeleton)
        captured = capsys.readouterr()
        assert "Test" in captured.out
        assert "c1" in captured.out
        assert "r1" in captured.out
        assert "p1" in captured.out


class TestPresentPodcastSkeleton:
    def test_output_contains_fields(self, capsys):
        skeleton = [{
            "topic": "Pod Topic",
            "core_argument": "Key argument",
            "evidence_chain": ["e1", "e2"],
            "tension_point": "tension",
            "key_quotes": ["quote"],
            "timestamp": "00:15",
            "relationships": ["rel"],
        }]
        _present_podcast_skeleton(skeleton)
        captured = capsys.readouterr()
        assert "Pod Topic" in captured.out
        assert "Key argument" in captured.out
        assert "e1" in captured.out
        assert "tension" in captured.out
        assert "quote" in captured.out
        assert "00:15" in captured.out


# ── run_skeletonize ───────────────────────────────────────────────────

class TestRunSkeletonize:
    def _setup(self, work_dir, content_type="book"):
        """Set up mocks and real checkpoint for run_skeletonize."""
        cfg = {"api": {"backend": "deepseek"}, "output": {"work_dir": work_dir}}
        fw = {
            "skeleton": {
                "methodology_prompt": "extract",
                "quality_rules": {"min_topics": 2, "require_claims": True},
            },
        }
        mgr = CheckpointManager(work_dir=work_dir)
        mgr.init_checkpoint("test-book", "/p/b.epub", "epub", content_type=content_type)
        return cfg, fw

    @patch("lumen.core.analyzer._load_framework")
    @patch("lumen.core.analyzer._create_analyzer")
    @patch("lumen.core.analyzer.load_config")
    def test_happy_path(self, mock_config, mock_create, mock_framework, sample_chunks, work_dir):
        cfg, fw = self._setup(work_dir)
        mock_config.return_value = cfg
        mock_framework.return_value = fw

        analyzer = MagicMock()
        analyzer.skeletonize.return_value = [
            {"topic": "Topic 1", "claims": ["c1"], "relationships": [], "prerequisites": []},
            {"topic": "Topic 2", "claims": ["c2"], "relationships": [], "prerequisites": []},
        ]
        mock_create.return_value = analyzer

        with patch.dict(os.environ, {'LUMEN_AUTO_APPROVE': '1'}, clear=True):
            skeleton = run_skeletonize("test-book", sample_chunks, content_type="book")
            assert len(skeleton) == 2
            assert skeleton[0]["topic"] == "Topic 1"

    @patch("lumen.core.analyzer._load_framework")
    @patch("lumen.core.analyzer._create_analyzer")
    @patch("lumen.core.analyzer.load_config")
    def test_podcast_content_type(self, mock_config, mock_create, mock_framework, sample_chunks, work_dir):
        cfg, fw = self._setup(work_dir, content_type="podcast")
        mock_config.return_value = cfg
        mock_framework.return_value = fw

        analyzer = MagicMock()
        analyzer.skeletonize.return_value = [
            {"topic": "P1", "core_argument": "arg", "evidence_chain": ["e1"], "tension_point": "", "timestamp": "00:05"},
        ]
        mock_create.return_value = analyzer

        with patch.dict(os.environ, {'LUMEN_AUTO_APPROVE': '1'}, clear=True):
            skeleton = run_skeletonize("test-book", sample_chunks, content_type="podcast")
            assert skeleton[0]["topic"] == "P1"
            assert skeleton[0]["core_argument"] == "arg"


# ── run_deep_read ─────────────────────────────────────────────────────

class TestRunDeepRead:
    @patch("lumen.core.analyzer._create_analyzer")
    @patch("lumen.core.analyzer.load_config")
    def test_happy_path(self, mock_config, mock_create, mock_analyzer, sample_chunks, sample_skeleton, work_dir):
        mock_config.return_value = {"api": {"backend": "deepseek"}, "output": {"work_dir": work_dir}}
        mock_create.return_value = mock_analyzer

        mgr = CheckpointManager(work_dir=work_dir)
        mgr.init_checkpoint("test-book", "/p/b.epub", "epub")

        with patch.dict(os.environ, {}, clear=True):
            results = run_deep_read("test-book", sample_chunks, sample_skeleton, content_type="book")
            assert len(results) == 2
            assert results[0]["chapter_id"] == "ch-01"

    @patch("lumen.core.analyzer._create_analyzer")
    @patch("lumen.core.analyzer.load_config")
    def test_empty_chunks(self, mock_config, mock_create, work_dir):
        mock_config.return_value = {"api": {"backend": "deepseek"}, "output": {"work_dir": work_dir}}
        mock_create.return_value = MagicMock()

        mgr = CheckpointManager(work_dir=work_dir)
        mgr.init_checkpoint("test-book", "/p/b.epub", "epub")
        results = run_deep_read("test-book", [], [], content_type="book")
        assert results == []


# ── run_synthesis ─────────────────────────────────────────────────────

class TestRunSynthesis:
    @patch("lumen.core.analyzer._create_analyzer")
    @patch("lumen.core.analyzer.load_config")
    def test_happy_path(self, mock_config, mock_create, mock_analyzer, work_dir):
        mock_config.return_value = {"api": {"backend": "deepseek"}, "output": {"work_dir": work_dir}}
        mock_create.return_value = mock_analyzer

        mgr = CheckpointManager(work_dir=work_dir)
        mgr.init_checkpoint("test-book", "/p/b.epub", "epub")

        with patch.dict(os.environ, {}, clear=True):
            result = run_synthesis("test-book", [{"chapter_id": "ch-01"}])
            assert result["book_summary"] == "Test summary."
            assert result["core_concepts"][0]["name"] == "Concept 1"

    @patch("lumen.core.analyzer._create_analyzer")
    @patch("lumen.core.analyzer.load_config")
    def test_empty_analyses(self, mock_config, mock_create, work_dir):
        mock_config.return_value = {"api": {"backend": "deepseek"}, "output": {"work_dir": work_dir}}
        mock_create.return_value.synthesize.return_value = {
            "book_summary": "", "core_concepts": [], "key_arguments": [],
        }

        mgr = CheckpointManager(work_dir=work_dir)
        mgr.init_checkpoint("test-book", "/p/b.epub", "epub")

        with patch.dict(os.environ, {}, clear=True):
            result = run_synthesis("test-book", [])
            assert "book_summary" in result
