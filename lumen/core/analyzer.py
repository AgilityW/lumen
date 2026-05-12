"""Analyzer factory + quality checks.

Routes to the correct LLM backend based on config. Runs quality verification
on skeleton output before presenting to the user.
"""

import functools
import json
import os
from typing import Any

import yaml

from lumen.core.config import load_config
from lumen.exceptions import ConfigError, UserInterrupt


@functools.lru_cache(maxsize=8)
def _load_framework(name: str = "technical") -> dict:
    """Load a framework YAML definition by name (e.g., 'podcast', 'book', 'technical')."""
    _BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    framework_paths = [
        os.path.join(_BASE, "frameworks", f"{name}.yaml"),
        os.path.join(_BASE, "frameworks", f"{name}.yml"),
        os.path.join("frameworks", f"{name}.yaml"),
        os.path.join("frameworks", f"{name}.yml"),
    ]
    for path in framework_paths:
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f) or {}

    # Fallback: try "default"
    default_paths = [
        os.path.join(_BASE, "frameworks", "default.yaml"),
        os.path.join(_BASE, "frameworks", "default.yml"),
        os.path.join("frameworks", "default.yaml"),
        os.path.join("frameworks", "default.yml"),
    ]
    for path in default_paths:
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f) or {}

    print(f"[WARN] Framework '{name}' not found. Using hardcoded defaults.")
    return {
        "skeleton": {
            "methodology_prompt": "Extract the core topics, claims, and relationships from this content.",
            "quality_rules": {"min_topics": 3, "require_claims": True},
        },
        "chapter": {
            "analysis_prompt": "Analyze this section anchored to the skeleton.",
            "output_schema": {},
        },
    }


def _detect_framework(content_type: str) -> str:
    """Map content type to framework file name."""
    from lumen.core.classifier import get_framework_name
    return get_framework_name(content_type)


def _create_analyzer(config: dict | None = None):
    """Factory: create the right Analyzer backend based on config."""
    if config is None:
        config = load_config()

    api_config = config.get("api", {})
    backend = api_config.get("backend", "deepseek")

    if backend == "deepseek":
        from lumen.backends.deepseek import DeepSeekAnalyzer
        ds = api_config.get("deepseek", {})
        api_key = (
            os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ds.get("api_key")
            or ""
        )
        if not api_key:
            raise ConfigError("DeepSeek API key not configured. Run `lumen init` or set DEEPSEEK_API_KEY.")
        return DeepSeekAnalyzer(
                api_key=api_key,
                model=ds.get("model", "deepseek-chat"),
                base_url=ds.get("base_url", "https://api.deepseek.com"),
            )

    if backend == "claude":
        from lumen.backends.claude_api import ClaudeAPIAnalyzer
        cl = api_config.get("claude", {})
        api_key = os.environ.get("CLAUDE_API_KEY") or cl.get("api_key") or ""
        if not api_key:
            raise ConfigError("Claude API key not configured. Run `lumen init` or set CLAUDE_API_KEY.")
        return ClaudeAPIAnalyzer(api_key=api_key, model=cl.get("model", "claude-sonnet-4-20250514"))

    raise ConfigError(f"Unknown backend '{backend}'. Supported: deepseek, claude")


def run_skeletonize(book_slug: str, chunks: list[dict], content_type: str = "unknown") -> list[dict]:
    """Run Phase 2 skeleton extraction with content-type-aware framework selection.

    3-pass pipeline:
       Pass 1 — Initial extraction
       Pass 1.5 — Adaptive archetype generation
       Pass 2 — Gap analysis (max 2 rounds)
       Pass 3 — Structural refinement (dedup, merge, reorder)
       GATE  — User review (approve/redo/quit)

    Returns the validated skeleton (list of topics).
    """
    config = load_config()

    # ── Slug resolution ────────────────────────────────────────────────
    from lumen.core.state import CheckpointManager
    work_dir = os.environ.get("LUMEN_WORK_DIR") or config.get("output", {}).get("work_dir", "output")
    _slug_mgr = CheckpointManager(work_dir=work_dir)
    resolved = _slug_mgr.resolve_slug(book_slug)
    if resolved != book_slug:
        print(f"[Phase 2] Slug auto-resolved: '{book_slug}' -> '{resolved}'")
        book_slug = resolved

    # ── Framework + Analyzer setup ────────────────────────────────────
    framework_name = _detect_framework(content_type)
    framework = _load_framework(framework_name)
    analyzer = _create_analyzer(config)

    print(f"[Phase 2] Content type: {content_type}")
    print(f"[Phase 2] Framework: {framework_name}")
    quality_rules = framework.get("skeleton", {}).get("quality_rules", {})

    # ── Pass 1: Initial extraction ────────────────────────────────────
    print("[Phase 2] Pass 1 — Initial skeleton extraction...")
    skeleton = analyzer.skeletonize(chunks, framework, content_type=content_type)
    quality_ok, quality_issues = _check_skeleton_quality(skeleton, quality_rules, content_type)
    skeleton = _ensure_skeleton_field_safety(skeleton, content_type)

    if not quality_ok:
        print(f"[Phase 2] Quality check failed: {'; '.join(quality_issues)}")
        print("[Phase 2] Retrying with adjusted prompt...")
        adjusted_framework = dict(framework)
        adjusted_framework["skeleton"] = dict(framework.get("skeleton", {}))
        adjusted_framework["skeleton"]["methodology_prompt"] = (
            "CRITICAL: Your previous extraction was incomplete. "
            + framework.get("skeleton", {}).get("methodology_prompt", "")
            + f"\n\nSpecifically fix these issues: {'; '.join(quality_issues)}"
            + "\n\nEnsure you extract at least 5 major topics with clear core arguments."
        )
        skeleton = analyzer.skeletonize(chunks, adjusted_framework, content_type=content_type)
        skeleton = _ensure_skeleton_field_safety(skeleton, content_type)

    print(f"[Phase 2] Pass 1 complete: {len(skeleton)} topics extracted.")
    for t in skeleton:
        print(f"  - {t.get('topic', '?')}")

    # ── Pass 1.5: Adaptive archetype generation ───────────────────────
    generic_archetypes = framework.get("skeleton", {}).get("coverage_archetypes", {})
    full_text = _build_full_text(chunks, max_chars=15000)

    adapt_result = analyzer.adapt_archetypes(
        skeleton=skeleton,
        raw_content=full_text,
        content_type=content_type,
    )
    domain_archetypes = adapt_result.get("domain_archetypes", {})
    if domain_archetypes:
        print(f"[Phase 2] Merging {len(domain_archetypes)} domain-specific + {len(generic_archetypes)} generic archetypes")
        coverage_archetypes = {**generic_archetypes, **domain_archetypes}
    else:
        coverage_archetypes = generic_archetypes

    # ── Pass 2 + 3: Gap analysis & refinement ─────────────────────────
    if coverage_archetypes:
        skeleton = _run_gap_analysis_and_refine(
            analyzer, skeleton, chunks, full_text,
            coverage_archetypes, content_type, quality_rules,
        )
    else:
        print("[Phase 2] No coverage archetypes. Skipping Pass 2 & 3.")

    # ── Save skeleton ──────────────────────────────────────────────────
    skeleton = _ensure_skeleton_field_safety(skeleton, content_type)
    quality_ok, quality_issues = _check_skeleton_quality(skeleton, quality_rules, content_type)
    if not quality_ok:
        print(f"[WARN] Final quality issues: {'; '.join(quality_issues)}")
        print("[WARN] User can request redo at review gate.")

    skeleton_path = _save_skeleton(skeleton, book_slug, work_dir)
    print(f"[Phase 2] Final skeleton: {len(skeleton)} topics.")

    # ── GATE: User review ──────────────────────────────────────────────
    decision, skeleton = _handle_skeleton_review_gate(
        skeleton, book_slug, framework, chunks, content_type, analyzer, skeleton_path,
    )

    if decision == "quit":
        raise UserInterrupt("Exiting at user request.")

    return skeleton


def _build_full_text(chunks: list[dict], max_chars: int = 0) -> str:
    """Concatenate chunk text for gap analysis, optionally truncated early."""
    full_text_parts = []
    est_len = 0
    for c in chunks:
        text = c.get("text", c) if isinstance(c, dict) else c
        source = c.get("source", "") if isinstance(c, dict) else ""
        if source:
            part = f"=== {source} ===\n{text}"
        else:
            part = text
        full_text_parts.append(part)
        est_len += len(part) + 2
        if max_chars and est_len >= max_chars:
            break
    full = "\n\n".join(full_text_parts)
    if max_chars:
        return full[:max_chars]
    return full


def _save_skeleton(skeleton: list[dict], book_slug: str, work_dir: str) -> str:
    """Save skeleton JSON and update checkpoint. Returns skeleton path."""
    from lumen.core.state import CheckpointManager
    analysis_dir = os.path.join(work_dir, book_slug, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    skeleton_path = os.path.join(analysis_dir, "skeleton.json")
    with open(skeleton_path, "w") as f:
        json.dump(skeleton, f, indent=2, ensure_ascii=False)
    manager = CheckpointManager(work_dir=work_dir)
    manager.update_artifacts(book_slug, {"skeleton": skeleton_path})
    print(f"[Phase 2] Saved to {skeleton_path}")
    return skeleton_path


def _handle_skeleton_review_gate(
    skeleton: list[dict],
    book_slug: str,
    framework: dict,
    chunks: list[dict],
    content_type: str,
    analyzer: Any,
    skeleton_path: str,
) -> tuple[str, list[dict]]:
    """Phase 2 GATE: present skeleton for user review.

    Returns (decision, final_skeleton):
      decision: "approve" | "redo" | "quit"
      final_skeleton: the (possibly regenerated) skeleton
    """
    from lumen.core.state import CheckpointManager
    work_dir = os.path.dirname(os.path.dirname(skeleton_path))

    book_title = book_slug.replace("-", " ").title()
    decision, feedback = present_skeleton_for_review(skeleton, book_title, content_type)

    if decision == "redo":
        print(f"[Phase 2] User requested redo: {feedback}")
        adjusted_framework = dict(framework)
        adj_skeleton = dict(framework.get("skeleton", {}))
        adj_skeleton["methodology_prompt"] = (
            "The user reviewed your previous extraction and provided this feedback:\n"
            f"{feedback}\n\n"
            + framework.get("skeleton", {}).get("methodology_prompt", "")
            + "\n\nAddress the user's feedback above all else."
        )
        adjusted_framework["skeleton"] = adj_skeleton
        skeleton = analyzer.skeletonize(chunks, adjusted_framework, content_type=content_type)
        with open(skeleton_path, "w") as f:
            json.dump(skeleton, f, indent=2, ensure_ascii=False)
        manager = CheckpointManager(work_dir=work_dir)
        manager.update_artifacts(book_slug, {"skeleton": skeleton_path})
        print(f"[Phase 2] Redone skeleton: {len(skeleton)} topics.")

    # Finalize checkpoint phase
    manager = CheckpointManager(work_dir=work_dir)
    manager.update_phase(book_slug, "gate_passed")
    print("[Phase 2] Gate passed. Ready for Phase 3.")
    return decision, skeleton


def _run_gap_analysis_and_refine(
    analyzer: Any,
    skeleton: list[dict],
    chunks: list[dict],
    full_text: str,
    archetypes: dict,
    content_type: str,
    quality_rules: dict,
) -> list[dict]:
    """Pass 2 (gap analysis) + Pass 3 (refinement) for coverage improvement."""
    all_new = []
    seen_topics = set()
    for t in skeleton:
        tname = t.get("topic", "")
        if tname:
            seen_topics.add(tname.lower()[:60])

    for round_idx in range(2):
        print(f"[Phase 2] Pass 2 — Gap analysis round {round_idx + 1}...")
        gap_result = analyzer.gap_analyze(
            skeleton=skeleton,
            archetypes=archetypes,
            raw_content=full_text[:12000],
            content_type=content_type,
        )
        new_topics_list: list[dict] = gap_result.get("new_topics", [])
        new_topics_list = [_fill_missing_fields(t, content_type) for t in new_topics_list]

        if not new_topics_list:
            print(f"[Phase 2] Round {round_idx + 1} found no gaps.")
            break

        fresh_topics = []
        for t in new_topics_list:
            tname = t.get("topic", "")
            if not tname:
                continue
            tname_lower = tname.lower()[:60]
            if tname_lower not in seen_topics:
                seen_topics.add(tname_lower)
                fresh_topics.append(t)
            else:
                print(f"  (skipping duplicate: {tname[:60]})")

        if fresh_topics:
            print(f"[Phase 2] Round {round_idx + 1} extracted {len(fresh_topics)} new topics:")
            for t in fresh_topics:
                print(f"  + {t.get('topic', '?')}")
            all_new.extend(fresh_topics)
        else:
            print(f"[Phase 2] Round {round_idx + 1} found no truly new topics.")
            if round_idx > 0:
                break

    if all_new:
        print(f"[Phase 2] Pass 3 — Structural refinement ({len(skeleton) + len(all_new)} topics)...")
        skeleton = analyzer.refine_skeleton(
            initial_skeleton=skeleton,
            new_topics=all_new,
            content_type=content_type,
        )
        print(f"[Phase 2] Pass 3 complete: {len(skeleton)} topics after refinement.")
    else:
        print("[Phase 2] No new topics from gap analysis. Skipping refinement.")

    return skeleton


def _check_skeleton_quality(skeleton: list[dict], rules: dict, content_type: str = "unknown") -> tuple[bool, list[str]]:
    """Check skeleton output against quality rules.

    Uses soft_max_topics as early warning and max_topics as absolute hard cap.
    Returns (passed, list_of_issues) — passed is True when no hard-cap violations exist.
    Content-type-aware quality checks.
    """
    issues = []
    hard_failure = False

    min_topics = rules.get("min_topics", 0)
    if min_topics and len(skeleton) < min_topics:
        issues.append(f"Only {len(skeleton)} topics (need at least {min_topics})")
        hard_failure = True

    max_topics = rules.get("max_topics", 99)
    soft_max_topics = rules.get("soft_max_topics", 0)
    if max_topics and len(skeleton) > max_topics:
        issues.append(f"{len(skeleton)} topics (absolute max {max_topics})")
        hard_failure = True
    elif soft_max_topics and len(skeleton) > soft_max_topics:
        issues.append(f"{len(skeleton)} topics exceeds soft cap ({soft_max_topics})")

    # Content-type-specific checks
    if content_type == "podcast":
        if rules.get("require_core_argument"):
            for topic in skeleton:
                if not topic.get("core_argument"):
                    tname = topic.get("topic", "?")
                    issues.append(f"Topic '{tname}' missing 'core_argument'")
                    hard_failure = True
                    break

        if rules.get("require_tension_point"):
            topics_with_tension = sum(1 for t in skeleton if t.get("tension_point"))
            if topics_with_tension < len(skeleton) * 0.5:
                issues.append(f"Only {topics_with_tension}/{len(skeleton)} topics have tension_point")
                hard_failure = True

        if rules.get("require_timestamp"):
            topics_with_ts = sum(1 for t in skeleton if t.get("timestamp"))
            if topics_with_ts < len(skeleton) * 0.5:
                issues.append(f"Only {topics_with_ts}/{len(skeleton)} topics have timestamps")
                hard_failure = True
    else:
        # Book/article/reference checks
        if rules.get("require_claims"):
            topics_with_claims = sum(1 for t in skeleton if isinstance(t.get("claims"), list) and len(t["claims"]) > 0)
            if topics_with_claims < len(skeleton) * 0.5:
                issues.append(f"Only {topics_with_claims}/{len(skeleton)} topics have non-empty claims")
                hard_failure = True

        if rules.get("require_relationships"):
            topics_with_rels = sum(
                1 for t in skeleton
                if isinstance(t.get("relationships"), list) and len(t["relationships"]) > 0
            )
            if topics_with_rels < len(skeleton) * 0.5:
                issues.append(f"Only {topics_with_rels}/{len(skeleton)} topics have explicit relationships")
                hard_failure = True

    return not hard_failure, issues


def _fill_missing_fields(topic: dict, content_type: str = "unknown") -> dict:
    """Ensure every topic has minimum required fields, filling defaults if missing.

    Prevents downstream crashes when the LLM returns partial structures.
    Returns the topic dict with guaranteed fields.
    """
    if content_type == "podcast":
        fields = {
            "topic": "",
            "core_argument": "No explicit core argument extracted.",
            "evidence_chain": [],
            "tension_point": "No tension point explicitly identified.",
            "key_quotes": [],
            "timestamp": "",
            "speaker": "",
            "relationships": [],
        }
    else:
        fields = {
            "topic": "",
            "description": "",
            "claims": [],
            "relationships": [],
            "prerequisites": [],
            "page_hints": [],
        }
    for key, default in fields.items():
        if key not in topic or topic[key] is None:
            topic[key] = default
        elif isinstance(topic[key], list) and not isinstance(default, list):
            topic[key] = default
        elif isinstance(topic[key], str) and not topic[key].strip():
            topic[key] = default if isinstance(default, str) else topic[key]
    return topic


def _ensure_skeleton_field_safety(skeleton: list[dict], content_type: str = "unknown") -> list[dict]:
    """Apply field-level safety fill to every topic in the skeleton."""
    return [_fill_missing_fields(t, content_type) for t in skeleton]


def _analyze_one_chunk(
    analyzer: Any,
    chunk_text: str,
    skeleton: list[dict],
    source: str,
    idx: int,
    framework: dict,
    content_type: str,
    chunk: dict,
) -> dict:
    """Analyze a single chunk — extracted for ThreadPoolExecutor use."""
    analysis = analyzer.analyze_chapter(
        chunk_text,
        context={
            "skeleton": skeleton,
            "chapter": {"title": source, "index": idx},
            "framework": framework,
            "content_type": content_type,
        },
    )
    analysis["chunk_id"] = chunk.get("id", f"chunk-{idx+1:04d}")
    analysis["source"] = source
    return analysis


def run_deep_read(book_slug: str, chunks: list[dict], skeleton: list[dict], content_type: str = "unknown") -> list[dict]:
    """Phase 3: Deep-read each chapter anchored to the frozen skeleton.

    Chapters are analyzed in parallel via ThreadPoolExecutor (max 4 workers)
    since each chunk is independent. Results are reordered by chunk index.

    Returns list of chapter analyses (failed chunks get error placeholder dicts).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    config = load_config()

    # Auto-resolve slug from checkpoint
    from lumen.core.state import CheckpointManager
    work_dir = os.environ.get("LUMEN_WORK_DIR") or config.get("output", {}).get("work_dir", "output")
    _slug_mgr = CheckpointManager(work_dir=work_dir)
    resolved = _slug_mgr.resolve_slug(book_slug)
    if resolved != book_slug:
        print(f"[Phase 3] Slug auto-resolved: '{book_slug}' -> '{resolved}'")
        book_slug = resolved

    # Use the correct framework for deep-read prompts too
    framework_name = _detect_framework(content_type)
    framework = _load_framework(framework_name)

    analyzer = _create_analyzer(config)

    checkpoint_dir = os.environ.get("LUMEN_WORK_DIR") or config.get("output", {}).get("work_dir", "output")
    analysis_dir = os.path.join(checkpoint_dir, book_slug, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # ── Mark deep-read in progress ────────────────────────────────────
    from lumen.core.state import CheckpointManager
    manager = CheckpointManager(work_dir=checkpoint_dir)
    manager.update_phase(book_slug, "deep-read")

    total = len(chunks)
    print(f"[Phase 3] Analyzing {total} chunks in parallel (max 4 workers)...")

    # Prepare tasks
    tasks: list[tuple[int, str, str, dict]] = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get("text", chunk) if isinstance(chunk, dict) else chunk
        source = chunk.get("source", f"Chunk {i+1}") if isinstance(chunk, dict) else f"Chunk {i+1}"
        tasks.append((i, str(chunk_text), source, chunk))

    # Run in parallel, collect results indexed by position
    indexed_results: dict[int, dict] = {}
    failed_ids: list[str] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for idx, chunk_text, source, chunk in tasks:
            future = pool.submit(
                _analyze_one_chunk,
                analyzer, chunk_text, skeleton, source, idx,
                framework, content_type, chunk,
            )
            futures[future] = (idx, source, chunk.get("id", f"chunk-{idx+1:04d}"))

        for future in as_completed(futures):
            idx, source, chunk_id = futures[future]
            try:
                result = future.result()
                indexed_results[idx] = result
                print(f"[Phase 3] Chunk {idx+1}/{total} complete.")
            except Exception as exc:
                print(f"[Phase 3] Chunk {idx+1}/{total} FAILED: {exc}")
                manager.add_error(book_slug, "deep-read", f"Chunk {source} failed: {exc}")
                indexed_results[idx] = {
                    "chunk_id": chunk_id,
                    "source": source,
                    "error": str(exc),
                }
                failed_ids.append(chunk_id)

    if failed_ids:
        print(f"[WARN] {len(failed_ids)} of {total} chunks failed: {', '.join(failed_ids)}")

    # Reorder by chunk index — every index is guaranteed present
    analyses = [indexed_results[i] for i in sorted(indexed_results)]

    # ── Single write at end (no O(n²) serialization) ──────────────────
    partial_path = os.path.join(analysis_dir, "chapter_analyses.json")
    with open(partial_path, "w") as f:
        json.dump(analyses, f, indent=2, ensure_ascii=False)

    print(f"[Phase 3] Completed {len(analyses)} analyses ({len(analyses) - len(failed_ids)} ok, {len(failed_ids)} failed).")
    return analyses

def run_synthesis(book_slug: str, analyses: list[dict]) -> dict[str, Any]:
    """Synthesize all chapter analyses at end of Phase 3."""
    config = load_config()
    analyzer = _create_analyzer(config)

    print("[Phase 3] Synthesizing all chapter analyses...")
    synthesis = analyzer.synthesize(analyses)

    base = os.environ.get("LUMEN_WORK_DIR") or config.get("output", {}).get("work_dir", "output")
    analysis_dir = os.path.join(base, book_slug, "analysis")
    synthesis_path = os.path.join(analysis_dir, "synthesis.json")
    with open(synthesis_path, "w") as f:
        json.dump(synthesis, f, indent=2, ensure_ascii=False)

    # Update checkpoint
    from lumen.core.state import CheckpointManager
    manager = CheckpointManager(work_dir=os.environ.get("LUMEN_WORK_DIR") or config.get("output", {}).get("work_dir", "output"))
    manager.update_artifacts(book_slug, {"synthesis": synthesis_path})
    manager.update_phase(book_slug, "digest")

    print(f"[Phase 3] Synthesis complete. Saved to {synthesis_path}")
    return synthesis


def present_skeleton_for_review(skeleton: list[dict], book_title: str, content_type: str = "unknown") -> tuple[str, str]:
    """Phase 2 GATE: Display skeleton to user for review.

    Adapts display format based on content type.
    Skips interactive prompt when LUMEN_AUTO_APPROVE is set.

    Returns (decision, feedback):
      decision: "approve" | "redo" | "quit"
      feedback: user's textual feedback (empty if not redo)
    """
    auto = os.environ.get("LUMEN_AUTO_APPROVE", "").strip().lower()
    if auto in ("1", "true", "yes"):
        print()
        print("=" * 60)
        print(f"  Skeleton Review — {book_title} (LUMEN_AUTO_APPROVE)")
        print("=" * 60)
        print(f"  Auto-approved: {len(skeleton)} topics.")
        print()
        return "approve", ""

    while True:
        print()
        print("=" * 60)
        print(f"  Skeleton Review — {book_title}")
        print(f"  Content type: {content_type}")
        print("=" * 60)

        if content_type == "podcast":
            _present_podcast_skeleton(skeleton)
        else:
            _present_book_skeleton(skeleton)

        print()
        print("---")
        print("Options: [a]pprove skeleton and continue | [r]edo with feedback | [q]uit")

        choice = input("\nYour choice [a/r/q]: ").strip().lower()

        if choice in ("a", "approve", ""):
            return "approve", ""
        elif choice in ("r", "redo"):
            print("\n--- Describe what you want changed (e.g. missing topics, wrong emphasis, reorder) ---")
            feedback = input("Feedback: ").strip()
            if not feedback:
                print("[INFO] Empty feedback — defaulting to 'Please improve overall quality and coverage.'")
                feedback = "Please improve overall quality and coverage."
            return "redo", feedback
        elif choice in ("q", "quit"):
            print("[INFO] User quit at skeleton review gate.")
            return "quit", ""
        else:
            print(f"[INFO] Unknown choice '{choice}'. Try again.")


def _present_podcast_skeleton(skeleton: list[dict]) -> None:
    """Display podcast skeleton with its specific schema fields."""
    for i, topic in enumerate(skeleton, 1):
        tname = topic.get("topic", f"Topic {i}")
        arg = topic.get("core_argument", "")
        evidence = topic.get("evidence_chain", [])
        tension = topic.get("tension_point", "")
        quotes = topic.get("key_quotes", [])
        ts = topic.get("timestamp", "")
        rels = topic.get("relationships", [])

        print(f"\n  {i}. {tname}" + (f"  [{ts}]" if ts else ""))
        if arg:
            print(f"     Argument: {arg[:100]}")
        if evidence:
            print(f"     Evidence: {'; '.join(evidence[:2])}")
        if tension:
            print(f"     Tension: {tension[:80]}")
        if quotes:
            print(f"     Quote: {quotes[0][:100]}")
        if rels:
            print(f"     Relations: {'; '.join(rels[:2])}")


def _present_book_skeleton(skeleton: list[dict]) -> None:
    """Display book skeleton with traditional schema fields."""
    for i, topic in enumerate(skeleton, 1):
        tname = topic.get("topic", f"Topic {i}")
        claims = topic.get("claims", [])
        rels = topic.get("relationships", [])
        prereqs = topic.get("prerequisites", [])

        print(f"\n  {i}. {tname}")
        if claims:
            print(f"     Claims: {'; '.join(claims[:3])}")
        if rels:
            print(f"     Relationships: {'; '.join(rels[:2])}")
        if prereqs:
            print(f"     Prerequisites: {'; '.join(prereqs[:2])}")
