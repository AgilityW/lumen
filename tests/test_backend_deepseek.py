"""Tests for DeepSeekAnalyzer — mock the HTTP layer via OpenAICompatBackend."""

from unittest.mock import patch, MagicMock

import pytest

from lumen.backends.deepseek import DeepSeekAnalyzer


@pytest.fixture
def analyzer():
    return DeepSeekAnalyzer(api_key="test-key", model="test-model", base_url="https://api.test.com")


def _mock_chat(content: str):
    """Return a mock chat_completion result."""
    return {"content": content, "finish_reason": "stop"}


class TestDeepSeekAnalyzer:
    def test_skeletonize_parses_json_array(self, analyzer):
        with patch.object(analyzer._client, "chat_completion", return_value=_mock_chat(
            '[{"topic": "T1", "claims": ["c1"]}, {"topic": "T2", "claims": ["c2"]}]'
        )):
            result = analyzer.skeletonize(
                chunks=[{"text": "content", "source": "ch1"}],
                framework={"skeleton": {"methodology_prompt": "extract", "quality_rules": {}}},
            )
            assert len(result) == 2
            assert result[0]["topic"] == "T1"
            assert result[1]["topic"] == "T2"

    def test_skeletonize_handles_dict_wrapper(self, analyzer):
        with patch.object(analyzer._client, "chat_completion", return_value=_mock_chat(
            '{"topics": [{"topic": "T1"}]}'
        )):
            result = analyzer.skeletonize(
                chunks=[{"text": "x"}],
                framework={"skeleton": {"methodology_prompt": "extract", "quality_rules": {}}},
            )
            assert len(result) == 1
            assert result[0]["topic"] == "T1"

    def test_skeletonize_retry_on_parse_failure(self, analyzer):
        responses = [
            _mock_chat("not valid json"),
            _mock_chat('[{"topic": "T1"}]'),
        ]
        with patch.object(analyzer._client, "chat_completion", side_effect=responses):
            result = analyzer.skeletonize(
                chunks=[{"text": "x"}],
                framework={"skeleton": {"methodology_prompt": "extract", "quality_rules": {}}},
            )
            assert len(result) == 1
            assert result[0]["topic"] == "T1"

    def test_skeletonize_returns_empty_on_all_failures(self, analyzer):
        with patch.object(analyzer._client, "chat_completion", return_value=_mock_chat("invalid")):
            result = analyzer.skeletonize(
                chunks=[{"text": "x"}],
                framework={"skeleton": {"methodology_prompt": "extract", "quality_rules": {}}},
            )
            assert result == []

    def test_skeletonize_normalizes_string_items(self, analyzer):
        with patch.object(analyzer._client, "chat_completion", return_value=_mock_chat(
            '["topic1", "topic2"]'
        )):
            result = analyzer.skeletonize(
                chunks=[{"text": "x"}],
                framework={"skeleton": {"methodology_prompt": "extract", "quality_rules": {}}},
            )
            assert len(result) == 2
            assert result[0]["topic"] == "topic1"
            assert result[1]["topic"] == "topic2"

    def test_analyze_chapter_returns_dict(self, analyzer):
        with patch.object(analyzer._client, "chat_completion", return_value=_mock_chat(
            '{"chapter_id": "ch1", "topics_covered": []}'
        )):
            result = analyzer.analyze_chapter(
                chunk="some text",
                context={"skeleton": [], "chapter": {"title": "Ch1"}, "content_type": "book", "framework": {}},
            )
            assert result["chapter_id"] == "ch1"

    def test_analyze_chapter_returns_empty_on_failure(self, analyzer):
        with patch.object(analyzer._client, "chat_completion", return_value=_mock_chat("invalid")):
            result = analyzer.analyze_chapter(
                chunk="text",
                context={"skeleton": [], "chapter": {}, "content_type": "book", "framework": {}},
            )
            assert result == {}

    def test_synthesize_returns_dict(self, analyzer):
        with patch.object(analyzer._client, "chat_completion", return_value=_mock_chat(
            '{"book_summary": "summary", "core_concepts": []}'
        )):
            result = analyzer.synthesize(analyses=[{"chapter_id": "ch1"}])
            assert result["book_summary"] == "summary"

    def test_synthesize_returns_error_on_failure(self, analyzer):
        with patch.object(analyzer._client, "chat_completion", return_value=_mock_chat("invalid")):
            result = analyzer.synthesize(analyses=[{"chapter_id": "ch1"}])
            assert "error" in result

    def test_gap_analyze_returns_dict(self, analyzer):
        with patch.object(analyzer._client, "chat_completion", return_value=_mock_chat(
            '{"new_topics": [], "gap_assessment": [], "reasoning": "ok"}'
        )):
            result = analyzer.gap_analyze(skeleton=[], archetypes={}, raw_content="text")
            assert result["reasoning"] == "ok"

    def test_refine_skeleton_returns_list(self, analyzer):
        with patch.object(analyzer._client, "chat_completion", return_value=_mock_chat(
            '[{"topic": "T1"}]'
        )):
            result = analyzer.refine_skeleton(initial_skeleton=[], new_topics=[])
            assert len(result) == 1

    def test_refine_skeleton_fallback_on_parse_failure(self, analyzer):
        with patch.object(analyzer._client, "chat_completion", return_value=_mock_chat("invalid")):
            result = analyzer.refine_skeleton(
                initial_skeleton=[{"topic": "T1"}],
                new_topics=[{"topic": "T2"}],
            )
            assert len(result) == 2

    def test_check_coverage_returns_list(self, analyzer):
        with patch.object(analyzer._client, "chat_completion", return_value=_mock_chat(
            '{"coverage": [{"archetype": "test", "status": "covered", "note": ""}]}'
        )):
            result = analyzer.check_coverage(skeleton=[], archetypes={})
            assert len(result) == 1

    def test_adapt_archetypes_returns_dict(self, analyzer):
        with patch.object(analyzer._client, "chat_completion", return_value=_mock_chat(
            '{"domain": "test", "domain_archetypes": {}, "reasoning": "ok"}'
        )):
            result = analyzer.adapt_archetypes(skeleton=[], raw_content="text")
            assert result["domain"] == "test"
