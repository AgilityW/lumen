"""Pipeline orchestration — init_config and sync_to_vault entry points.

These are thin wrappers called by the CLI. Heavy lifting lives in the
individual phase modules (parser, chunker, analyzer, renderer).
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

import yaml

from lumen.exceptions import ConfigError

log = logging.getLogger("lumen.pipeline")


def _update_dotenv(env_path: str, updates: dict[str, str]) -> None:
    """Update or append KEY=value pairs in a .env file without a parser dependency."""
    existing: dict[str, str] = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    existing[k.strip()] = v.strip()

    existing.update(updates)
    with open(env_path, "w") as f:
        for k, v in existing.items():
            f.write(f"{k}={v}\n")


def init_config() -> None:
    """One-time setup: prompt user for backend, vault path, API keys.

    API keys are written to .env (not config.yaml) to prevent accidental
    credential leaks via git. Non-secret config goes to config.yaml.
    """
    from lumen.core.config import default_config, find_config, invalidate_config_cache
    existing = find_config()
    if existing:
        print(f"[INFO] Config already exists at {existing}")
        override = input("Overwrite? [y/N] ").strip().lower()
        if override != "y":
            print("[INFO] Keeping existing config.")
            return

    config = default_config()

    print("Lumen Setup")
    print("=" * 40)

    # Backend selection
    backend = input("Backend [deepseek/claude] (default: deepseek): ").strip().lower()
    if backend:
        config["api"]["backend"] = backend

    env_updates: dict[str, str] = {}
    if config["api"]["backend"] == "claude":
        claude_key = input("Claude API key (or press Enter to skip): ").strip()
        if claude_key:
            env_updates["CLAUDE_API_KEY"] = claude_key
        claude_model = input("Claude model (default: claude-sonnet-4-20250514): ").strip()
        if claude_model:
            config["api"]["claude"]["model"] = claude_model
    else:
        ds_key = input("DeepSeek API key (or press Enter to skip): ").strip()
        if ds_key:
            env_updates["DEEPSEEK_API_KEY"] = ds_key
        ds_model = input("DeepSeek model (default: deepseek-chat): ").strip()
        if ds_model:
            config["api"]["deepseek"]["model"] = ds_model

    # Vault path
    vault_path = input("Obsidian vault path (or press Enter to skip): ").strip()
    if vault_path:
        config["vault"]["path"] = vault_path

    # Framework defaults (list available)
    frameworks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frameworks")
    if os.path.isdir(frameworks_dir):
        available = sorted(f for f in os.listdir(frameworks_dir) if f.endswith((".yaml", ".yml")))
        if available:
            names = ", ".join(f.replace(".yaml", "").replace(".yml", "") for f in available)
            print(f"  Available frameworks: {names}")
            fw = input("Default framework (default: technical): ").strip()
            if fw:
                config["framework"]["default"] = fw

    # api_key is read ONLY from env (DEEPSEEK_API_KEY, CLAUDE_API_KEY, etc.)
    target = os.path.join(os.getcwd(), "config.yaml")
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    invalidate_config_cache()
    print(f"[OK] Config written to {target}")

    # Write API keys to .env
    if env_updates:
        env_target = os.path.join(os.getcwd(), ".env")
        _update_dotenv(env_target, env_updates)
        print(f"[OK] API keys written to {env_target}")
        print("     Keys are loaded from .env at runtime. Keep this file out of version control.")
    else:
        print("[INFO] No API keys provided. Set DEEPSEEK_API_KEY or CLAUDE_API_KEY in .env later.")


def sync_to_vault() -> None:
    """Persist runtime data from latest checkpoint to Obsidian vault."""
    from lumen.core.config import load_config
    config = load_config()
    if not config.get("vault", {}).get("path"):
        raise ConfigError("Run `lumen init` first to configure.")

    vault_path = config.get("vault", {}).get("path", "")
    if not vault_path:
        raise ConfigError("vault.path not configured.")

    work_dir = os.environ.get("LUMEN_WORK_DIR") or config.get("output", {}).get("work_dir", "output")
    if not os.path.isdir(work_dir):
        raise ConfigError("No working directory found. Start a book first.")

    # Locate the most recent checkpoint
    from lumen.core.state import CheckpointManager
    manager = CheckpointManager(work_dir=work_dir)

    checkpoints = []
    for item in sorted(Path(work_dir).iterdir()):
        cp = manager.load_checkpoint(item.name)
        if cp and cp.get("phase") in ("digest", "complete"):
            checkpoints.append(cp)

    if not checkpoints:
        print("[INFO] No completed books to sync.")
        return

    # Digest all completed books
    from lumen.renderers.mindmap import MindmapRenderer
    from lumen.renderers.obsidian import ObsidianRenderer

    obsidian = ObsidianRenderer(vault_path)
    mindmap = MindmapRenderer()

    for cp in checkpoints:
        slug = cp["book_slug"]
        synthesis_path = cp.get("artifacts", {}).get("synthesis", "")
        if not synthesis_path or not os.path.exists(synthesis_path):
            print(f"[WARN] No synthesis found for '{slug}', skipping.")
            continue

        with open(synthesis_path) as f:
            synthesis = json.load(f)

        obsidian.render(slug, synthesis)
        mm_content = mindmap.render(synthesis)
        obsidian.write_mindmap(slug, mm_content)
        manager.sync_timestamp(slug)
        print(f"[OK] Synced '{slug}' to vault.")


def ingest_book(book_path: str) -> dict:
    """Phase 1: Ingestion & Init.

    Parses the book, chunks it, builds structure.json, and initializes
    checkpoint. Called conversationally by Codex (no CLI command).

    Returns the parsed book dict with chunks and structure, ready for Phase 2.
    """
    from lumen.core.chunker import chunk_book
    from lumen.core.classifier import classify
    from lumen.core.parser import build_structure, parse_book
    from lumen.core.state import CheckpointManager

    # Resolve config for output directory
    # LUMEN_WORK_DIR env var overrides config
    work_dir = os.environ.get("LUMEN_WORK_DIR") or _load_config().get("output", {}).get("work_dir", "output")

    # 1. Parse
    print(f"[Phase 1] Parsing {book_path}...")
    parsed = parse_book(book_path)

    # 2. Classify content type
    print("[Phase 1] Classifying content type...")
    content_type = classify(parsed.get("text", ""))
    parsed["content_type"] = content_type
    print(f"[Phase 1] Detected content type: {content_type}")

    # 3. Chunk (content-type-aware)
    print("[Phase 1] Chunking...")
    parsed = chunk_book(parsed)

    # 4. Build structure
    structure = build_structure(parsed)
    print(f"[Phase 1] Extracted {structure['total_chapters']} chapters, "
          f"{structure['total_chunks']} chunks.")

    # 5. Book slug from filename
    slug = _to_slug(Path(book_path).stem)

    # 6. Write structure.json
    output_dir = Path(work_dir) / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    structure_path = output_dir / "structure.json"
    with open(structure_path, "w") as f:
        json.dump(structure, f, indent=2, ensure_ascii=False)

    # 7. Init checkpoint
    manager = CheckpointManager(work_dir=work_dir)
    manager.init_checkpoint(slug, book_path, parsed.get("format", ""), content_type=content_type)
    manager.update_phase(slug, "skeletonize")

    # Store artifact paths
    manager.update_artifacts(slug, {
        "structure": str(structure_path),
    })

    print(f"[Phase 1] Complete. Output in {output_dir}/")
    print("[Phase 1] Checkpoint phase: skeletonize")
    parsed["slug"] = slug
    parsed["content_type"] = content_type
    return parsed


def _to_slug(name: str) -> str:
    """Convert a filename to a slug suitable for directory names."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", slug)
    slug = slug.strip("-")
    return slug or "book"


def _load_config() -> dict:
    from lumen.core.config import load_config
    return load_config()


def run_digestion(book_slug: str) -> None:
    """Phase 4: Digestion — render synthesis to Obsidian notes + Mermaid mind map.

    Called after Phase 3 synthesis is complete. Uses the checkpoint artifact
    paths to locate the synthesis output.
    """
    config = _load_config()
    work_dir = os.environ.get("LUMEN_WORK_DIR") or config.get("output", {}).get("work_dir", "output")
    vault_path = config.get("vault", {}).get("path", "")
    book_dir = config.get("vault", {}).get("book_dir", "Books")

    from lumen.core.state import CheckpointManager

    # Auto-resolve slug from checkpoint
    _slug_mgr = CheckpointManager(work_dir=work_dir)
    resolved = _slug_mgr.resolve_slug(book_slug)
    if resolved != book_slug:
        print(f"[Phase 4] Slug auto-resolved: '{book_slug}' -> '{resolved}'")
        book_slug = resolved
    from lumen.renderers.mindmap import MindmapRenderer
    from lumen.renderers.obsidian import ObsidianRenderer

    manager = CheckpointManager(work_dir=work_dir)
    cp = manager.load_checkpoint(book_slug)
    if cp is None:
        print(f"[ERROR] No checkpoint found for '{book_slug}'", file=sys.stderr)
        return

    synthesis_path = cp.get("artifacts", {}).get("synthesis", "")
    if not synthesis_path or not os.path.exists(synthesis_path):
        print(f"[ERROR] No synthesis found for '{book_slug}'. Run Phase 3 first.", file=sys.stderr)
        return

    with open(synthesis_path) as f:
        synthesis = json.load(f)

    print("[Phase 4] Rendering to Obsidian vault...")

    # Render notes
    obsidian = ObsidianRenderer(vault_path=vault_path, book_dir=book_dir)
    note_path = obsidian.render(book_slug, synthesis)
    print(f"[Phase 4] Book note: {note_path}")

    # Render mind map
    mindmap = MindmapRenderer()
    mm_content = mindmap.render(synthesis)
    obsidian.write_mindmap(book_slug, mm_content)
    print("[Phase 4] Mind map embedded in book note.")

    # Render WeChat HTML
    from lumen.renderers.html import WeChatRenderer
    wc = WeChatRenderer(book_title=book_slug)
    html_content = wc.render(synthesis)
    html_dir = os.path.join(work_dir, book_slug)
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(html_dir, f"{book_slug}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[Phase 4] WeChat HTML: {html_path}")

    # Update checkpoint
    manager.update_phase(book_slug, "complete")
    print(f"[Phase 4] Complete. Book '{book_slug}' is ready in vault.")
    print("[Phase 4] Run `lumen sync` to ensure all data is persisted.")


def _check_book_path(book_path: str) -> str:
    """Resolve book path, with fallback to output directory slug resolution."""
    path = os.path.abspath(book_path)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Book not found: {book_path}")


def run_full_pipeline(book_path: str) -> None:
    """Run all 4 phases on a book: ingest → skeletonize → deep-read → digest.

    This is the main entry point for the `lumen run` CLI command.
    """
    path = _check_book_path(book_path)

    log.info("[Pipeline] Starting full pipeline for %s", book_path)

    # Phase 1: Ingest
    parsed = ingest_book(path)

    slug = parsed["slug"]
    chunks = parsed.get("chunks", [])
    content_type = parsed.get("content_type", "unknown")

    if not chunks:
        log.error("No chunks extracted from '%s'. Cannot proceed.", book_path)
        return

    # Phase 2: Skeletonize
    from lumen.core.analyzer import run_skeletonize
    skeleton = run_skeletonize(slug, chunks, content_type=content_type)

    if not skeleton:
        log.error("Skeleton extraction yielded no topics for '%s'. Cannot proceed.", slug)
        return

    # Phase 3: Deep Read + Synthesis
    from lumen.core.analyzer import run_deep_read, run_synthesis
    analyses = run_deep_read(slug, chunks, skeleton, content_type=content_type)
    synthesis = run_synthesis(slug, analyses)

    if not synthesis or "error" in synthesis:
        log.warning("Synthesis may be incomplete for '%s'.", slug)

    # Phase 4: Digestion
    run_digestion(slug)

    log.info("[OK] Pipeline complete for '%s'. Run `lumen sync` to sync to vault.", slug)
