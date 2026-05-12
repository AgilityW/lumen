"""DeepSeek Analyzer — default backend for Autonomous mode.

Content-type aware: adapts extraction strategy and output schema
based on whether input is a podcast, book, article, or reference.
"""

import json
import sys
from typing import Any

from lumen.backends.base import BaseAnalyzer
from lumen.backends.openai_compat import OpenAICompatBackend


class DeepSeekAnalyzer(BaseAnalyzer):
    """Analyzer backed by DeepSeek API (OpenAI-compatible)."""

    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com"):
        self._client = OpenAICompatBackend(api_key=api_key, base_url=base_url, model=model)

    def _chat(self, system: str, user: str, response_format: dict | None = None) -> str:
        result = self._client.chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=response_format or {"type": "json_object"},
        )
        return result["content"]

    def _parse(self, content: str) -> dict | list | None:
        return self._client.parse_json_response(content)

    def _chat_parse(self, system: str, user: str) -> dict | list | None:
        return self._parse(self._chat(system, user))

    @staticmethod
    def _build_chunk_text(chunks: list, max_chars: int = 4000) -> str:
        parts: list[str] = []
        for i, c in enumerate(chunks):
            text = c.get("text", c) if isinstance(c, dict) else c
            source = c.get("source", f"Section {i+1}") if isinstance(c, dict) else f"Section {i+1}"
            if len(text) > max_chars:
                text = text[:max_chars] + "\n[...truncated...]"
            parts.append(f"=== {source} ===\n\n{text}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _build_quality_text(rules: dict) -> str:
        lines: list[str] = []
        checks: list[tuple[str, str]] = [
            ("min_topics", "Include at least {} major topics."),
            ("max_topics", "Do not exceed {} topics."),
            ("require_core_argument", "Each topic must have a clear core_argument."),
            ("require_evidence_chain", "Each topic should have evidence_chain with 2-3 reasoning steps."),
            ("require_tension_point", "Each topic should include a tension_point (counter-argument or trade-off)."),
            ("require_timestamp", "Each topic should include approximate timestamp from the episode."),
            ("require_claims", "Each topic must have at least one clear claim or argument."),
            ("require_relationships", "Explicitly describe relationships between topics."),
            ("require_prerequisites", "Each topic should include prerequisite knowledge."),
        ]
        for key, template in checks:
            val = rules.get(key)
            if val:
                lines.append(f"- {template.format(val) if '{}' in template else template}")
        return "\n".join(lines)

    @staticmethod
    def _build_skeleton_text(skeleton: list[dict], max_len: int = 120) -> str:
        return "\n".join(
            f"- {t.get('topic', '?')}: {t.get('core_argument', '')[:max_len]}"
            for t in skeleton
        )

    # ── Core methods ─────────────────────────────────────────────────────────

    def skeletonize(self, chunks: list, framework: dict, content_type: str = "unknown") -> list[dict]:
        methodology = framework.get("skeleton", {}).get("methodology_prompt", "")
        quality_rules = framework.get("skeleton", {}).get("quality_rules", {})
        output_schema = framework.get("skeleton", {}).get("output_schema", {})

        chunk_text = self._build_chunk_text(chunks)
        quality_rule_text = self._build_quality_text(quality_rules)

        if content_type == "podcast":
            system_msg = (
                "You are a podcast analysis engine. Your task is to extract the "
                "guest's core arguments, design decisions, and technical insights "
                "from a podcast transcript. Focus on the substance, not the conversation flow.\n\n"
                "Output ONLY valid JSON. No markdown, no explanation, no commentary."
            )
        else:
            system_msg = (
                "You are a book/podcast analysis engine. Your task is to extract the "
                "underlying knowledge structure from a book or podcast transcript. "
                "Focus on concepts, claims, and their relationships.\n\n"
                "Output ONLY valid JSON. No markdown, no explanation, no commentary."
            )

        if content_type == "podcast" and output_schema:
            output_spec = (
                'Output the skeleton as a JSON array of objects, each with:\n'
                '  - "topic": string (concise topic name)\n'
                '  - "core_argument": string (the guest\'s central claim)\n'
                '  - "evidence_chain": array of strings (supporting reasoning steps)\n'
                '  - "tension_point": string (counter-argument, trade-off, or skeptical angle)\n'
                '  - "key_quotes": array of strings (1-2 verbatim quotes with timestamps if available)\n'
                '  - "timestamp": string (approximate time range, e.g. "09:42~20:24")\n'
                '  - "speaker": string (primary speaker for this topic — usually the guest)\n'
                '  - "relationships": array of strings (how this relates to other topics, e.g. "depends-on: X")\n'
            )
        else:
            output_spec = (
                'Output the skeleton as a JSON array of objects, each with:\n'
                '  - "topic": string (concise topic name)\n'
                '  - "description": string (1-2 sentence explanation)\n'
                '  - "claims": array of strings (key claims the author makes)\n'
                '  - "relationships": array of strings (how this topic relates to others)\n'
                '  - "prerequisites": array of strings (knowledge needed to understand this topic)\n'
                '  - "page_hints": array of strings (any section references found)\n'
            )

        user_msg = (
            f"{methodology}\n\n"
            f"**Quality Requirements:**\n{quality_rule_text}\n"
            f"**Full Content:**\n\n{chunk_text}\n\n"
            f"{output_spec}"
        )

        content = self._chat(system_msg, user_msg)
        parsed = self._parse(content)

        if parsed is None and content:
            print("[WARN] Skeleton response was not valid JSON. Retrying with stricter prompt...")
            content = self._chat(
                "You are a JSON-only analysis engine. Output ONLY a raw JSON array. No markdown.",
                f"Analyze this content and output a JSON array of topics:\n\n{chunk_text}",
            )
            parsed = self._parse(content)

        if parsed is None:
            print("[ERROR] Could not parse skeleton as JSON even after retry.", file=sys.stderr)
            return []

        if isinstance(parsed, dict):
            parsed = parsed.get("topics", parsed.get("skeleton", parsed))

        if not isinstance(parsed, list):
            return []

        normalized: list[dict] = []
        for item in parsed:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, str):
                normalized.append({"topic": item, "core_argument": "", "evidence_chain": [], "tension_point": ""})
            else:
                normalized.append({"topic": str(item), "core_argument": ""})
        return normalized

    def analyze_chapter(self, chunk: str, context: dict) -> dict[str, Any]:
        skeleton = context.get("skeleton", [])
        chapter_info = context.get("chapter", {})
        content_type = context.get("content_type", "unknown")

        if content_type == "podcast":
            system_msg = (
                "You are a podcast analysis engine performing deep reading. "
                "You have the frozen skeleton of the episode. Analyze how this segment "
                "contributes to the skeleton. Output ONLY valid JSON."
            )
        else:
            system_msg = (
                "You are a book analysis engine performing deep reading. "
                "You have the frozen skeleton of the book. Analyze how this chapter "
                "contributes to the skeleton. Output ONLY valid JSON."
            )

        skeleton_text = "\n".join(
            f"- {t.get('topic', '?')}: {t.get('core_argument') or ', '.join(t.get('claims', [])[:1]) or '?'}"
            for t in skeleton[:10]
        )

        chunk_content = chunk[:8000] if len(chunk) > 8000 else chunk

        if content_type == "podcast":
            output_spec = (
                'Output JSON with:\n'
                '- "segment_id": string\n'
                '- "timestamp_range": string (time range of this segment)\n'
                '- "speakers": array of strings\n'
                '- "topics_covered": array of {topic, reasoning_chain, verbatim_quotes, technical_details, speaker}\n'
                '- "new_topics": array of new topics introduced\n'
                '- "tensions": array of contradictions or refinements'
            )
        else:
            output_spec = (
                'Output JSON with:\n'
                '- "chapter_id": string\n'
                '- "topics_covered": array of {topic, key_insights, technical_details, code_references}\n'
                '- "new_concepts": array of new concepts introduced\n'
                '- "contradictions": array of refinements to skeleton claims'
            )

        user_msg = (
            f"**Frozen Skeleton (top-level topics):**\n{skeleton_text}\n\n"
            f"**Segment:** {chapter_info.get('title', 'Unknown')}\n"
            f"**Content:**\n{chunk_content}\n\n"
            f"{output_spec}"
        )

        parsed = self._chat_parse(system_msg, user_msg)
        return parsed if isinstance(parsed, dict) else {}

    def synthesize(self, analyses: list[dict]) -> dict[str, Any]:
        system_msg = (
            "You are a synthesis engine. Combine all chapter analyses into "
            "a unified understanding of the content. Output ONLY valid JSON."
        )

        analyses_text = json.dumps(analyses, indent=2, ensure_ascii=False)[:12000]

        user_msg = (
            f"**All Chapter Analyses:**\n{analyses_text}\n\n"
            f"Synthesize into JSON with:\n"
            f'- "book_summary": 2-3 paragraph overview\n'
            f'- "core_concepts": array of {{name, definition, importance}} (extract at least 6-8 concepts)\n'
            f'- "key_arguments": array of author main arguments\n'
            f'- "relationship_map": array of {{from, to, type}} (depends-on/conflicts/extends)\n'
            f'- "reading_notes": array of strings (actionable notes for the reader)'
        )

        parsed = self._chat_parse(system_msg, user_msg)
        return parsed if isinstance(parsed, dict) else {"error": "parse failed"}

    # ── Multi-pass skeleton methods ──────────────────────────────────────────

    def gap_analyze(
        self,
        skeleton: list[dict],
        archetypes: dict,
        raw_content: str = "",
        content_type: str = "unknown",
    ) -> dict:
        archetype_prompts: list[str] = []
        for key, arch in archetypes.items():
            label = arch.get("description", key)
            importance = arch.get("importance", "medium")
            check = arch.get("check_prompt", "")
            expected = arch.get("expected_topics", [])
            expected_text = f"  Expected topic pattern: {'; '.join(expected)}" if expected else ""
            archetype_prompts.append(f"[{importance}] {label}\n  Check: {check}\n{expected_text}")

        archetype_text = "\n\n".join(archetype_prompts)
        skeleton_text = self._build_skeleton_text(skeleton)

        if content_type == "podcast":
            system_msg = (
                "You are a podcast analysis engine performing gap analysis. "
                "You have a current skeleton and a list of archetypes that should "
                "be covered. Determine which archetypes are missing or under-covered, "
                "then extract the missing topics from the full transcript.\n\n"
                "Output ONLY valid JSON. No markdown, no explanation."
            )
        else:
            system_msg = (
                "You are a content analysis engine performing gap analysis. "
                "You have a current skeleton and coverage archetypes. "
                "Find missing topics and extract them from the full content.\n\n"
                "Output ONLY valid JSON."
            )

        key_desc_pairs: list[str] = []
        for key, arch in archetypes.items():
            label = arch.get("description", key)
            importance = arch.get("importance", "medium")
            key_desc_pairs.append(f"  {key}: ({importance}) {label}")
        key_desc_text = "\n".join(key_desc_pairs)

        user_msg = (
            f"**Current skeleton topics:**\n{skeleton_text}\n\n"
            f"**Coverage archetypes to check (use the KEY name in gap_assessment):**\n{archetype_text}\n\n"
            f"**Key-to-description mapping:**\n{key_desc_text}\n\n"
            f"**Full content:**\n{raw_content[:28000]}\n\n"
            "For each archetype, assess whether the current skeleton adequately covers it.\n"
            "- If COVERED: mark as covered, no action needed.\n"
            "- If PARTIAL: the skeleton touches on it but lacks depth. Extract exactly 1 "
            "new topic that adds the missing depth or dimension.\n"
            "- If MISSING: the archetype is entirely absent. Extract exactly 1-2 concrete new topics "
            "that fully address this gap.\n\n"
            "IMPORTANT RULES:\n"
            "1. Use the KEY name (not description) in gap_assessment 'archetype' field.\n"
            "2. Extract new topics for BOTH PARTIAL and MISSING archetypes.\n"
            "3. Do NOT extract topics that overlap with existing topics (check carefully).\n"
            "4. CRITICAL: Only extract a new topic if the full content actually contains it — "
            "do not fabricate. If the content doesn't cover the archetype, leave 'missing' with no new topic.\n"
            "5. Fewer high-quality new topics is better.\n\n"
            "Output JSON: an object with:\n"
            '- "gap_assessment": array of {"archetype": str, "status": str, "note": str}\n'
            '- "new_topics": array of topic objects (same schema as the skeleton)\n'
            '- "reasoning": str (brief justification)'
        )

        parsed = self._chat_parse(system_msg, user_msg)
        if not isinstance(parsed, dict):
            return {"new_topics": [], "gap_assessment": [], "reasoning": "parse failed"}

        gap_assessment = parsed.get("gap_assessment", [])
        new_topics_data = parsed.get("new_topics", [])
        if not isinstance(new_topics_data, list):
            new_topics_data = []

        print(f"[Gap Analysis] Assessed {len(gap_assessment)} archetypes")
        for g in gap_assessment:
            print(f"  {g.get('archetype', '?')}: {g.get('status', '?')}")
        print(f"[Gap Analysis] New topics extracted: {len(new_topics_data)}")

        return {
            "new_topics": new_topics_data,
            "gap_assessment": gap_assessment if isinstance(gap_assessment, list) else [],
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

        if content_type == "podcast":
            system_msg = (
                "You are a podcast skeleton refinement engine. "
                "You have an initial skeleton plus newly extracted topics from gap analysis. "
                "Merge, deduplicate, and restructure into the final cohesive skeleton.\n\n"
                "Output ONLY valid JSON. No markdown, no explanation."
            )
        else:
            system_msg = (
                "You are a skeleton refinement engine. "
                "Merge and deduplicate topics from initial extraction and gap analysis. "
                "Output ONLY valid JSON."
            )

        user_msg = (
            f"**Combined topics (initial + gap-extracted):**\n{topics_text}\n\n"
            "Merge these topics into a CONCISE final skeleton (8-12 topics):\n"
            "1. Aggressively deduplicate — merge topics covering the same concept into ONE. "
            "For example, if 'memory wall' appears in multiple topics, merge them into a single "
            "authoritative topic.\n"
            "2. Each output topic MUST have a non-empty 'core_argument' field. If a topic "
            "lacks a core_argument, merge it into a similar topic that has one.\n"
            "3. Combine evidence_chains across merged topics — keep the strongest points.\n"
            "4. Merge key_quotes — keep the single most impactful quote per topic.\n"
            "5. Establish relationships between topics (depends-on / conflicts-with / extends / supports).\n"
            "6. Order by logical flow, not by transcript sequence.\n"
            "7. CRITICAL: Do NOT exceed 12 topics. Every topic must have a substantive core_argument.\n\n"
            "Output the final skeleton as a JSON array of topic objects with the standard schema:\n"
            '- "topic": str\n'
            '- "core_argument": str (REQUIRED — non-empty)\n'
            '- "evidence_chain": array of strings\n'
            '- "tension_point": str (REQUIRED — non-empty)\n'
            '- "key_quotes": array of strings\n'
            '- "timestamp": str\n'
            '- "speaker": str\n'
            '- "relationships": array of strings'
        )

        parsed = self._chat_parse(system_msg, user_msg)
        if parsed is None:
            return all_topics

        if isinstance(parsed, dict):
            parsed = parsed.get("skeleton", parsed.get("topics", parsed))

        return parsed if isinstance(parsed, list) else all_topics

    def check_coverage(self, skeleton: list[dict], archetypes: dict) -> list[dict]:
        topics_text = self._build_skeleton_text(skeleton, max_len=100)

        archetype_list: list[str] = []
        for key, arch in archetypes.items():
            importance = arch.get("importance", "medium")
            desc = arch.get("description", key)
            check = arch.get("check_prompt", "")
            expected = arch.get("expected_topics", [])
            expected_str = "; ".join(expected) if expected else "(not specified)"
            archetype_list.append(f"[{importance}] {desc}\n  Expected: {expected_str}\n  Check: {check}")

        archetype_text = "\n\n".join(archetype_list)

        system_msg = (
            "You are a coverage verification engine. Given a skeleton and a set of "
            "archetypes that should be covered, determine which archetypes are "
            "adequately covered and which are still missing or under-covered.\n\n"
            "Output ONLY valid JSON. No markdown, no explanation."
        )

        user_msg = (
            f"**Current skeleton topics:**\n{topics_text}\n\n"
            f"**Coverage archetypes to verify:**\n{archetype_text}\n\n"
            "For each archetype, assess if the skeleton adequately covers it.\n\n"
            'Output JSON:\n'
            '{"coverage": [{"archetype": str, "status": "covered"|"partial"|"missing", "note": str}]}'
        )

        parsed = self._chat_parse(system_msg, user_msg)
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

        if content_type in ("book", "article", "reference"):
            source_label = "book" if content_type == "book" else "article" if content_type == "article" else "reference material"
            role_label = "the author"
            audience_label = "reader"
        else:
            source_label = "podcast transcript"
            role_label = "the guest"
            audience_label = "listener"

        system_msg = (
            f"You are a {source_label} gap archetype generator. Your job is NOT to "
            f"describe what the skeleton already covers — instead, scan the raw "
            f"{source_label} for important topics the skeleton MISSED.\n\n"
            f"Think like a skeptical editor: what did {role_label} discuss that the "
            f"skeleton left out? Generate archetypes that target these gaps.\n\n"
            "Output ONLY valid JSON. No markdown, no explanation."
        )

        user_msg = (
            f"**Extracted skeleton topics (what we ALREADY have):**\n{topics_text}\n\n"
            f"**Raw {source_label} excerpt (scan for MISSED content):**\n{raw_content[:12000]}\n\n"
            "Your mission: Generate 3-5 domain-specific coverage archetypes that "
            "identify IMPORTANT TOPICS IN THE RAW CONTENT that the skeleton "
            "does NOT cover.\n\n"
            "RULES:\n"
            "1. FIRST scan the raw content for any major arguments, design decisions, "
            "or technical insights that appear in the content but are NOT captured "
            "in the skeleton above.\n"
            "2. For each finding, create an archetype that targets that gap.\n"
            "3. If the skeleton is comprehensive and the raw content adds nothing "
            "truly new, generate at most 1-2 archetypes focused on depth (e.g., "
            f"'are there deeper quantitative derivations in the {source_label} not in skeleton?').\n"
            "4. Each archetype must have a specific CHECK PROMPT that references "
            f"the raw {source_label} — something like 'Check sections on X for {role_label}'s "
            "argument about Y that the skeleton misses.'\n"
            f"5. IMPORTANCE: Use this rule — if the gap is a MAJOR topic the "
            f"{audience_label} should know, mark it 'critical'. If it's a useful but "
            "secondary addition, 'high'. If it's a nuance or detail, 'medium'.\n\n"
            "Output JSON:\n"
            '{\n'
            '  "domain": str (short domain label, e.g. "chip_architecture", "inference_economics", "model_training"),\n'
            '  "domain_archetypes": {\n'
            '    "archetype_key_1": {\n'
            '      "description": str,\n'
            f'      "check_prompt": str (specific {source_label} check for gap content),\n'
            '      "importance": "critical"|"high"|"medium"\n'
            '    },\n'
            '    ... (max 5 archetypes)\n'
            '  },\n'
            '  "reasoning": str (explain specifically what content in the raw content '
            'motivated each archetype)\n'
            '}'
        )

        parsed = self._chat_parse(system_msg, user_msg)
        if not isinstance(parsed, dict):
            return {"domain": "unknown", "domain_archetypes": {}, "reasoning": "parse failed"}

        domain = parsed.get("domain", "unknown")
        domain_archetypes = parsed.get("domain_archetypes", {})
        reasoning = parsed.get("reasoning", "")

        if not isinstance(domain_archetypes, dict):
            domain_archetypes = {}

        print(f"[Adaptive Archetypes] Detected domain: {domain}")
        for key, arch in domain_archetypes.items():
            imp = arch.get("importance", "?") if isinstance(arch, dict) else "?"
            desc = arch.get("description", key) if isinstance(arch, dict) else str(arch)[:80]
            print(f"  + [{imp}] {desc}")
        print(f"[Adaptive Archetypes] Reasoning: {reasoning[:200]}")

        return {
            "domain": domain,
            "domain_archetypes": domain_archetypes,
            "reasoning": reasoning,
        }
