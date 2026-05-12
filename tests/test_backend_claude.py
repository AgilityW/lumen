"""Tests for ClaudeAPIAnalyzer — mock the HTTP layer."""

from unittest.mock import patch, MagicMock

import pytest

from lumen.backends.claude_api import ClaudeAPIAnalyzer
from lumen.exceptions import APIError


@pytest.fixture
def analyzer():
    return ClaudeAPIAnalyzer(api_key="test-key", model="test-model")


class TestClaudeAPIAnalyzer:
    def test_skeletonize_returns_list(self, analyzer):
        with patch.object(analyzer, "_call", return_value='[{"topic": "T1"}]'):
            result = analyzer.skeletonize(
                chunks=[{"text": "content"}],
                framework={"skeleton": {"methodology_prompt": "extract", "quality_rules": {}}},
            )
            assert len(result) == 1
            assert result[0]["topic"] == "T1"

    def test_skeletonize_handles_dict_wrapper(self, analyzer):
        with patch.object(analyzer, "_call", return_value='{"topics": [{"topic": "T1"}]}'):
            result = analyzer.skeletonize(
                chunks=[{"text": "x"}],
                framework={"skeleton": {"methodology_prompt": "extract", "quality_rules": {}}},
            )
            assert len(result) == 1

    def test_skeletonize_returns_empty_on_failure(self, analyzer):
        with patch.object(analyzer, "_call", return_value="invalid"):
            result = analyzer.skeletonize(
                chunks=[{"text": "x"}],
                framework={"skeleton": {"methodology_prompt": "extract", "quality_rules": {}}},
            )
            assert result == []

    def test_analyze_chapter_returns_dict(self, analyzer):
        with patch.object(analyzer, "_call", return_value='{"chapter_id": "ch1"}'):
            result = analyzer.analyze_chapter(
                chunk="text",
                context={"skeleton": [], "chapter": {"title": "Ch1"}, "content_type": "book", "framework": {}},
            )
            assert result["chapter_id"] == "ch1"

    def test_synthesize_returns_dict(self, analyzer):
        with patch.object(analyzer, "_call", return_value='{"book_summary": "s"}'):
            result = analyzer.synthesize(analyses=[])
            assert result["book_summary"] == "s"

    def test_gap_analyze_returns_dict(self, analyzer):
        with patch.object(analyzer, "_call", return_value='{"new_topics": [], "gap_assessment": [], "reasoning": "ok"}'):
            result = analyzer.gap_analyze(skeleton=[], archetypes={}, raw_content="text")
            assert result["reasoning"] == "ok"

    def test_refine_skeleton_returns_list(self, analyzer):
        with patch.object(analyzer, "_call", return_value='[{"topic": "T1"}]'):
            result = analyzer.refine_skeleton(initial_skeleton=[], new_topics=[])
            assert len(result) == 1

    def test_check_coverage_returns_list(self, analyzer):
        with patch.object(analyzer, "_call", return_value='{"coverage": [{"archetype": "t", "status": "covered", "note": ""}]}'):
            result = analyzer.check_coverage(skeleton=[], archetypes={})
            assert len(result) == 1

    def test_adapt_archetypes_returns_dict(self, analyzer):
        with patch.object(analyzer, "_call", return_value='{"domain": "test", "domain_archetypes": {}, "reasoning": "ok"}'):
            result = analyzer.adapt_archetypes(skeleton=[], raw_content="text")
            assert result["domain"] == "test"

    def test_call_success(self, analyzer):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"content": [{"text": "ok"}]}

        with patch("requests.post", return_value=mock_resp):
            result = analyzer._call(system="sys", messages=[{"role": "user", "content": "hi"}])
            assert result == "ok"

    def test_call_unauthorized_raises(self, analyzer):
        mock_resp = MagicMock()
        mock_resp.status_code = 401

        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(APIError, match="Invalid Claude API key"):
                analyzer._call(system="sys", messages=[])
