"""End-to-end integration test for the full 4-phase pipeline.

Mocks the LLM backend at the _create_analyzer level, uses real parsers
and chunkers for a small markdown test book.
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from lumen.core.pipeline import run_full_pipeline


@pytest.fixture
def work_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def test_book():
    """Create a small markdown test book."""
    with tempfile.TemporaryDirectory() as tmp:
        book_path = os.path.join(tmp, "test-book.md")
        content = """# Test Book

## Chapter 1: Introduction
This is the first chapter. It covers basic concepts and foundational knowledge.

## Chapter 2: Core Concepts
This chapter dives deeper into the main ideas. It explains the key mechanisms.

## Chapter 3: Advanced Topics
The final chapter covers advanced applications and future directions.
"""
        with open(book_path, "w") as f:
            f.write(content)
        yield book_path


@pytest.fixture
def mock_analyzer():
    """Mock analyzer that returns canned responses for all 7 BaseAnalyzer methods."""
    mock = MagicMock()

    mock.skeletonize.return_value = [
        {"topic": "Basic Concepts", "claims": ["Foundational knowledge is important"], "relationships": [], "prerequisites": []},
        {"topic": "Core Mechanisms", "claims": ["The main mechanism works via X"], "relationships": ["Basic Concepts -> Core Mechanisms"], "prerequisites": ["Basic Concepts"]},
        {"topic": "Advanced Applications", "claims": ["Future directions include Y"], "relationships": ["Core Mechanisms -> Advanced Applications"], "prerequisites": ["Core Mechanisms"]},
    ]

    mock.adapt_archetypes.return_value = {
        "domain": "test_domain",
        "domain_archetypes": {},
        "reasoning": "No gaps found.",
    }

    mock.gap_analyze.return_value = {"gaps": [], "new_topics": []}

    mock.refine_skeleton.return_value = [
        {"topic": "Basic Concepts", "claims": ["Foundational knowledge is important"], "relationships": [], "prerequisites": []},
        {"topic": "Core Mechanisms", "claims": ["The main mechanism works via X"], "relationships": ["Basic Concepts -> Core Mechanisms"], "prerequisites": ["Basic Concepts"]},
        {"topic": "Advanced Applications", "claims": ["Future directions include Y"], "relationships": ["Core Mechanisms -> Advanced Applications"], "prerequisites": ["Core Mechanisms"]},
    ]

    mock.check_coverage.return_value = [{"archetype": "test", "status": "covered", "note": "ok"}]

    mock.analyze_chapter.return_value = {
        "chapter_id": "ch-01",
        "topics_covered": [{"topic": "Basic Concepts", "key_insights": ["Insight 1"]}],
    }

    mock.synthesize.return_value = {
        "book_summary": "A test book summary.",
        "core_concepts": [
            {"name": "Concept 1", "definition": "Definition 1", "relationships": []},
        ],
        "key_arguments": ["Argument 1"],
        "relationship_map": [],
    }

    return mock


@pytest.fixture
def config(work_dir):
    return {
        "api": {"backend": "deepseek"},
        "output": {"work_dir": work_dir},
        "vault": {"path": work_dir, "book_dir": "Books"},
    }


@pytest.fixture
def framework():
    return {
        "skeleton": {
            "methodology_prompt": "Extract topics.",
            "quality_rules": {"min_topics": 2, "require_claims": True},
            "output_schema": {},
            "coverage_archetypes": {},
        },
    }


class TestEndToEndPipeline:
    """End-to-end integration test for the full 4-phase pipeline."""

    @patch("lumen.core.analyzer._create_analyzer")
    @patch("lumen.core.analyzer.load_config")
    @patch("lumen.core.analyzer._load_framework")
    @patch("lumen.core.pipeline._load_config")
    def test_full_pipeline_completes_all_phases(
        self,
        mock_pipeline_config,
        mock_load_framework,
        mock_analyzer_config,
        mock_create_analyzer,
        test_book,
        mock_analyzer,
        config,
        framework,
        work_dir,
    ):
        """Run the full pipeline with a mock LLM backend and verify all outputs."""
        mock_create_analyzer.return_value = mock_analyzer
        mock_analyzer_config.return_value = config
        mock_load_framework.return_value = framework
        mock_pipeline_config.return_value = config

        with patch.dict(os.environ, {"LUMEN_AUTO_APPROVE": "1", "LUMEN_WORK_DIR": work_dir}, clear=True):
            run_full_pipeline(test_book)

        # --- Verify all output artifacts exist ---

        # 1. Checkpoint
        cp_path = os.path.join(work_dir, "test-book", ".checkpoint.json")
        assert os.path.exists(cp_path), f"Checkpoint missing: {cp_path}"
        with open(cp_path) as f:
            cp = json.load(f)
        assert cp["phase"] == "complete", f"Expected phase=complete, got {cp['phase']}"

        # 2. Skeleton
        sk_path = os.path.join(work_dir, "test-book", "analysis", "skeleton.json")
        assert os.path.exists(sk_path), f"Skeleton missing: {sk_path}"
        with open(sk_path) as f:
            skeleton = json.load(f)
        assert len(skeleton) == 3
        assert skeleton[0]["topic"] == "Basic Concepts"

        # 3. Chapter analyses
        an_path = os.path.join(work_dir, "test-book", "analysis", "chapter_analyses.json")
        assert os.path.exists(an_path), f"Analyses missing: {an_path}"

        # 4. Synthesis
        sy_path = os.path.join(work_dir, "test-book", "analysis", "synthesis.json")
        assert os.path.exists(sy_path), f"Synthesis missing: {sy_path}"
        with open(sy_path) as f:
            synthesis = json.load(f)
        assert synthesis["book_summary"] == "A test book summary."
        assert len(synthesis["core_concepts"]) == 1

        # 5. WeChat HTML
        html_path = os.path.join(work_dir, "test-book", "test-book.html")
        assert os.path.exists(html_path), f"HTML missing: {html_path}"
        with open(html_path) as f:
            html = f.read()
        assert "test-book" in html
        assert "A test book summary." in html
