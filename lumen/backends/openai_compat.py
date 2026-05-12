"""OpenAI-compatible API backend base class.

Shared by DeepSeek, OpenAI, and any API that follows the OpenAI chat
completions format.
"""

import json
import os
import time
from typing import Any

import requests

from lumen.exceptions import APIError


class OpenAICompatBackend:
    """Base for any OpenAI-compatible chat API."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 120):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat_completion(
        self,
        messages: list[dict],
        response_format: dict | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Call the chat completions endpoint with retry + exponential backoff."""
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if response_format:
            payload["response_format"] = response_format

        last_error = None
        for attempt in range(max_retries):
            try:
                # Temporarily clear proxy env vars that requests may pick up
                # even when explicit proxies={"http": None, "https": None} is set.
                # SOCKS proxies in particular cause failures if pysocks is not installed.
                _proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "SOCKS_PROXY",
                               "http_proxy", "https_proxy", "all_proxy", "socks_proxy"]
                _saved = {}
                for _v in _proxy_vars:
                    if _v in os.environ:
                        _saved[_v] = os.environ.pop(_v)
                try:
                    resp = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                finally:
                    # Restore proxy env vars
                    os.environ.update(_saved)
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]
                    content = choice["message"]["content"]
                    return {"content": content, "finish_reason": choice.get("finish_reason", "stop")}

                if resp.status_code == 401:
                    raise APIError("Invalid API key. Check your config.")
                if resp.status_code == 429:
                    wait = min(2 ** attempt, 30)
                    print(f"[WARN] Rate limited. Retrying in {wait}s...")
                    time.sleep(wait)
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    continue

                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"

            except requests.exceptions.Timeout:
                wait = min(2 ** attempt, 30)
                print(f"[WARN] API timeout. Retrying in {wait}s...")
                time.sleep(wait)
                last_error = "timeout"
                continue
            except requests.exceptions.ConnectionError as e:
                wait = min(2 ** attempt, 30)
                print(f"[WARN] Connection error. Retrying in {wait}s...")
                time.sleep(wait)
                last_error = f"connection: {e}"
                continue
            except Exception as e:
                last_error = str(e)
                break

        raise APIError(f"API call failed after {max_retries} retries: {last_error}")

    def parse_json_response(self, content: str) -> dict | list | None:
        """Parse JSON from LLM response. Handles markdown code fences."""
        content = content.strip()
        # Strip markdown ```json ... ``` fences
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [ln for ln in lines if not ln.startswith("```")]
            content = "\n".join(lines).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
