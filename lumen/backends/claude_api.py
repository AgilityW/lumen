"""Claude API backend — optional backend using Anthropic's Messages API."""

import json
import time
from typing import Any

import requests

from lumen.backends.base import BaseAnalyzer
from lumen.exceptions import APIError


class ClaudeAPIAnalyzer(BaseAnalyzer):
    """Analyzer backed by Claude API."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model

    def _call(self, system: str, messages: list[dict], max_retries: int = 3) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": 8192,
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["content"][0]["text"]
                if resp.status_code == 401:
                    raise APIError("Invalid Claude API key.")
                if resp.status_code == 429:
                    wait = min(2 ** attempt, 30)
                    time.sleep(wait)
                    last_error = f"rate limit: {resp.text[:200]}"
                    continue
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                wait = min(2 ** attempt, 30)
                time.sleep(wait)
                last_error = str(e)

        raise APIError(f"Claude API failed after {max_retries} retries: {last_error}")

    def _parse_json(self, content: str) -> dict | list | None:
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [ln for ln in lines if not ln.startswith("```")]
            content = "\n".join(lines).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    def skeletonize(self, chunks: list, framework: dict, content_type: str = "unknown") -> list[dict]:
        methodology = framework.get("skeleton", {}).get("methodology_prompt", "")
        quality_rules = framework.get("skeleton", {}).get("quality_rules", {})

        chunk_parts = []
        for i, c in enumerate(chunks):
            text = c.get("text", c) if isinstance(c, dict) else c
            source = c.get("source", f"Section {i+1}") if isinstance(c, dict) else f"Section {i+1}"
            if len(text) > 4000:
                text = text[:4000] + "\n[...truncated...]"
            chunk_parts.append(f"=== {source} ===\n\n{text}")
        chunk_text = "\n\n---\n\n".join(chunk_parts)

        qr = ""
        if quality_rules.get("min_topics"):
            qr += f"- At least {quality_rules['min_topics']} major topics.\n"
        if quality_rules.get("require_claims"):
            qr += "- Each topic has at least one claim.\n"
        if quality_rules.get("require_relationships"):
            qr += "- Explicit relationships between topics.\n"

        if content_type == "podcast":
            system_msg = (
                "You are a podcast analysis engine. Extract the guest's core arguments "
                "and technical insights. Output ONLY valid JSON."
            )
        else:
            system_msg = (
                "You are a book skeleton extraction engine. Output ONLY valid JSON."
            )

        content = self._call(
            system=system_msg,
            messages=[{
                "role": "user",
                "content": (
                    f"{methodology}\n\n"
                    f"Quality rules:\n{qr}\n\n"
                    f"Book content (first segments):\n{chunk_text}\n\n"
                    f"Output JSON array of topics with: topic, claims[], relationships[], prerequisites[]"
                ),
            }],
        )
        parsed = self._parse_json(content)
        if isinstance(parsed, dict):
            parsed = parsed.get("topics", parsed.get("skeleton", parsed))
        return parsed if isinstance(parsed, list) else []

    def analyze_chapter(self, chunk: str, context: dict) -> dict[str, Any]:
        skeleton = context.get("skeleton", [])
        ch = context.get("chapter", {})
        sk_text = "\n".join(
            f"- {t.get('topic', '?')}: {', '.join(t.get('claims', []))}"
            for t in skeleton[:10]
        )
        content = self._call(
            system="You are a chapter analyzer. Output ONLY valid JSON.",
            messages=[{
                "role": "user",
                "content": (
                    f"Skeleton:\n{sk_text}\n\n"
                    f"Chapter: {ch.get('title', '?')}\n\n{chunk[:8000]}\n\n"
                    f"Output: {{chapter_id, topics_covered, new_concepts, contradictions}}"
                ),
            }],
        )
        parsed = self._parse_json(content)
        return parsed if isinstance(parsed, dict) else {}

    def synthesize(self, analyses: list[dict]) -> dict[str, Any]:
        text = json.dumps(analyses, indent=2)[:12000]
        content = self._call(
            system="You are a synthesis engine. Output ONLY valid JSON.",
            messages=[{
                "role": "user",
                "content": (
                    f"Analyses:\n{text}\n\n"
                    f"Output: {{book_summary, core_concepts[], key_arguments[], "
                    f"relationship_map[], reading_notes[]}}"
                ),
            }],
        )
        parsed = self._parse_json(content)
        return parsed if isinstance(parsed, dict) else {}

    # ── Multi-pass skeleton methods ──────────────────────────────────────

    def gap_analyze(
        self,
        skeleton: list[dict],
        archetypes: dict,
        raw_content: str = "",
        content_type: str = "unknown",
    ) -> dict:
        archetype_prompts = []
        for key, arch in archetypes.items():
            label = arch.get("description", key)
            importance = arch.get("importance", "medium")
            check = arch.get("check_prompt", "")
            expected = arch.get("expected_topics", [])
            expected_text = f"  Expected topic pattern: {'; '.join(expected)}" if expected else ""
            archetype_prompts.append(f"[{importance}] {label}\n  Check: {check}\n{expected_text}")

        archetype_text = "\n\n".join(archetype_prompts)

        skeleton_text = "\n".join(
            f"- {t.get('topic', '?')}: {t.get('core_argument', '')[:120]}"
            for t in skeleton
        )

        if content_type == "podcast":
            system_msg = (
                "You are a podcast gap analysis engine. "
                "Compare the current skeleton against required archetypes. "
                "Output ONLY valid JSON."
            )
        else:
            system_msg = (
                "You are a gap analysis engine. "
                "Find missing topics and extract them. Output ONLY valid JSON."
            )

        user_msg = (
            f"**Current skeleton topics:**\n{skeleton_text}\n\n"
            f"**Coverage archetypes to check:**\n{archetype_text}\n\n"
            f"**Full content:**\n{raw_content[:28000]}\n\n"
            "For each archetype, assess: COVERED | PARTIAL | MISSING.\n"
            "Extract new topics for PARTIAL or MISSING archetypes only.\n\n"
            "Output JSON:\n"
            '{"gap_assessment": [{"archetype": str, "status": str, "note": str}],\n'
            ' "new_topics": [topic objects with the same schema as skeleton],\n'
            ' "reasoning": str}'
        )

        content = self._call(system=system_msg, messages=[{"role": "user", "content": user_msg}])
        parsed = self._parse_json(content)
        if not isinstance(parsed, dict):
            return {"new_topics": [], "gap_assessment": [], "reasoning": "parse failed"}
        return {
            "new_topics": parsed.get("new_topics", []),
            "gap_assessment": parsed.get("gap_assessment", []),
            "reasoning": parsed.get("reasoning", ""),
        }

    def refine_skeleton(
        self,
        initial_skeleton: list[dict],
        new_topics: list[dict],
        content_type: str = "unknown",
    ) -> list[dict]:
        all_topics = initial_skeleton + new_topics
        topics_text = json.dumps(all_topics, indent=2, ensure_ascii=False)

        system_msg = "You are a skeleton refinement engine. Merge, deduplicate, and restructure. Output ONLY valid JSON."

        user_msg = (
            f"**Combined topics (initial + gap-extracted):**\n{topics_text}\n\n"
            "Merge these into a CONCISE final skeleton (8-12 topics):\n"
            "1. Aggressively deduplicate.\n"
            "2. Each output topic MUST have a non-empty core_argument.\n"
            "3. Establish relationships between topics.\n"
            "4. Order by logical flow.\n\n"
            "Output JSON array of topic objects with standard schema."
        )

        content = self._call(system=system_msg, messages=[{"role": "user", "content": user_msg}])
        parsed = self._parse_json(content)
        if parsed is None:
            return all_topics
        if isinstance(parsed, dict):
            parsed = parsed.get("skeleton", parsed.get("topics", parsed))
        return parsed if isinstance(parsed, list) else all_topics

    def check_coverage(self, skeleton: list[dict], archetypes: dict) -> list[dict]:
        topics_text = "\n".join(
            f"- {t.get('topic', '?')}: {t.get('core_argument', '')[:100]}"
            for t in skeleton
        )

        archetype_list = []
        for key, arch in archetypes.items():
            importance = arch.get("importance", "medium")
            desc = arch.get("description", key)
            archetype_list.append(f"[{importance}] {desc}")

        archetype_text = "\n\n".join(archetype_list)

        system_msg = "You are a coverage verification engine. Output ONLY valid JSON."

        user_msg = (
            f"**Current skeleton:**\n{topics_text}\n\n"
            f"**Coverage archetypes:**\n{archetype_text}\n\n"
            "For each archetype, assess: covered | partial | missing.\n\n"
            'Output: {"coverage": [{"archetype": str, "status": str, "note": str}]}'
        )

        content = self._call(system=system_msg, messages=[{"role": "user", "content": user_msg}])
        parsed = self._parse_json(content)
        if isinstance(parsed, dict):
            return parsed.get("coverage", [])
        return []

    def adapt_archetypes(
        self,
        skeleton: list[dict],
        raw_content: str = "",
        content_type: str = "unknown",
    ) -> dict:
        topics_text = "\n".join(
            f"- {t.get('topic', '?')}: {t.get('core_argument', '')[:150]}"
            for t in skeleton
        )

        if content_type == "podcast":
            source_label = "podcast transcript"
        elif content_type in ("book", "article", "reference"):
            source_label = content_type
        else:
            source_label = "source material"

        system_msg = (
            f"You are a {source_label} gap archetype generator. "
            "Scan the raw content for topics the skeleton missed. "
            "Output ONLY valid JSON."
        )

        user_msg = (
            f"**Extracted skeleton topics:**\n{topics_text}\n\n"
            f"**Raw {source_label} excerpt:**\n{raw_content[:12000]}\n\n"
            "Generate 3-5 coverage archetypes targeting topics the skeleton missed.\n\n"
            'Output: {"domain": str, "domain_archetypes": {key: '
            '{"description": str, "check_prompt": str, "importance": str}}, "reasoning": str}'
        )

        content = self._call(system=system_msg, messages=[{"role": "user", "content": user_msg}])
        parsed = self._parse_json(content)
        if not isinstance(parsed, dict):
            return {"domain": "unknown", "domain_archetypes": {}, "reasoning": "parse failed"}
        return {
            "domain": parsed.get("domain", "unknown"),
            "domain_archetypes": parsed.get("domain_archetypes", {}),
            "reasoning": parsed.get("reasoning", ""),
        }
