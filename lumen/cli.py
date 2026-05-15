"""CLI entry point for Lumen.

Usage:
    lumen init              One-time setup
    lumen run <book>        Run full pipeline on a book
    lumen sync              Persist runtime data to Obsidian vault
    lumen status            Dashboard: progress, pending items, reading suggestions
"""

import argparse
import sys

from lumen.core.state import CheckpointManager
from lumen.exceptions import LumenError, UserInterrupt


def cmd_init(args: argparse.Namespace) -> int:
    """One-time setup: API key, vault path, generate config."""
    from lumen.core.pipeline import init_config

    init_config()
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    """Persist runtime data from checkpoint to Obsidian vault."""
    from lumen.core.pipeline import sync_to_vault

    sync_to_vault()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run full pipeline on a book: ingest → skeletonize → deep-read → digest."""
    from lumen.core.pipeline import run_full_pipeline
    run_full_pipeline(args.book)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show progress dashboard across all books."""
    import json as _json
    manager = CheckpointManager()
    if args.json:
        data = manager.build_dashboard_json()
        print(_json.dumps(data, indent=2, ensure_ascii=False))
    else:
        status_data = manager.build_dashboard()
        print(status_data)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lumen",
        description="Lumen — your book deconstruction engine. "
                    "Drop in a book, get structured notes and a mind map.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # lumen init
    init_p = sub.add_parser("init", help="One-time setup: API key, vault path, generate config")
    init_p.set_defaults(func=cmd_init)

    # lumen run
    run_p = sub.add_parser("run", help="Run full pipeline on a book: PDF/EPUB/MD")
    run_p.add_argument("book", help="Path to the book file")
    run_p.set_defaults(func=cmd_run)

    # lumen sync
    sync_p = sub.add_parser("sync", help="Persist runtime data to Obsidian vault")
    sync_p.set_defaults(func=cmd_sync)

    # lumen status
    status_p = sub.add_parser("status", help="Dashboard: progress, pending items, reading suggestions")
    status_p.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    status_p.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except UserInterrupt:
        return 0
    except LumenError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        return 1
