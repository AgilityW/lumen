"""Tests for OpenAI-compatible backend — mock the HTTP layer."""

import json
from unittest.mock import patch, MagicMock

import pytest

from lumen.backends.openai_compat import OpenAICompatBackend
from lumen.exceptions import APIError


class TestOpenAICompatBackend:
    @pytest.fixture
    def backend(self):
        return OpenAICompatBackend(api_key="test-key", base_url="https://api.test.com", model="test-model")

    def test_chat_completion_success(self, backend):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"result": "ok"}'}, "finish_reason": "stop"}]
        }

        with patch("requests.post", return_value=mock_response):
            result = backend.chat_completion(messages=[{"role": "user", "content": "hello"}])
            assert result["content"] == '{"result": "ok"}'
            assert result["finish_reason"] == "stop"

    def test_chat_completion_unauthorized(self, backend):
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(APIError, match="Invalid API key"):
                backend.chat_completion(messages=[{"role": "user", "content": "hello"}])

    def test_chat_completion_rate_limit_then_success(self, backend):
        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429
        rate_limit_resp.text = "rate limit"

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }

        with patch("requests.post", side_effect=[rate_limit_resp, success_resp]):
            result = backend.chat_completion(messages=[{"role": "user", "content": "hello"}])
            assert result["content"] == "ok"

    def test_chat_completion_all_retries_fail(self, backend):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "server error"

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(APIError, match="API call failed"):
                backend.chat_completion(messages=[{"role": "user", "content": "hello"}])

    def test_chat_completion_timeout_retry(self, backend):
        from requests.exceptions import Timeout

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }

        with patch("requests.post", side_effect=[Timeout(), success_resp]):
            result = backend.chat_completion(messages=[{"role": "user", "content": "hello"}])
            assert result["content"] == "ok"

    def test_parse_json_simple(self, backend):
        assert backend.parse_json_response('{"key": "value"}') == {"key": "value"}

    def test_parse_json_with_fences(self, backend):
        raw = '```json\n{"key": "value"}\n```'
        assert backend.parse_json_response(raw) == {"key": "value"}

    def test_parse_json_invalid(self, backend):
        assert backend.parse_json_response("not json") is None

    def test_parse_json_array(self, backend):
        assert backend.parse_json_response('[{"a": 1}]') == [{"a": 1}]

    def test_response_format_passed_to_request(self, backend):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            backend.chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                response_format={"type": "json_object"},
            )
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["response_format"] == {"type": "json_object"}
