"""Checkpoint manager — serializable JSON state for resume across sessions.

Every phase produces checkpoint updates. Checkpoints are the runtime memory
of the pipeline: progress, discussion history, cross-links, user preferences.
"""

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT = {
    "book_slug": "",
    "book_path": "",
    "book_format": "",
    "content_type": "unknown",
    "phase": "init",
    "chapters": [],
    "artifacts": {},
    "errors": [],
    "created_at": "",
    "updated_at": "",
    "version": 1,
}


class CheckpointManager:
    """Load, save, and query checkpoints for all books."""

    def __init__(self, work_dir: str | None = None):
        self.work_dir = work_dir or os.environ.get(
            "LUMEN_WORK_DIR",
            os.path.join(os.getcwd(), "output"),
        )
        self._slug_cache: dict[str, str] | None = None

    def _checkpoint_path(self, book_slug: str) -> Path:
        return Path(self.work_dir) / book_slug / ".checkpoint.json"

    def init_checkpoint(self, book_slug: str, book_path: str, book_format: str, content_type: str = "unknown") -> dict:
        """Create a new checkpoint for a book."""
        now = datetime.now(timezone.utc).isoformat()
        checkpoint = dict(DEFAULT_CHECKPOINT)
        checkpoint.update({
            "book_slug": book_slug,
            "book_path": book_path,
            "book_format": book_format,
            "content_type": content_type,
            "phase": "init",
            "created_at": now,
            "updated_at": now,
        })
        self._write(checkpoint, book_slug)
        return checkpoint

    def load_checkpoint(self, book_slug: str) -> dict | None:
        """Load checkpoint for a book, or None if not found."""
        path = self._checkpoint_path(book_slug)
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    def update_phase(self, book_slug: str, phase: str) -> dict:
        """Advance pipeline phase in checkpoint."""
        cp = self.load_checkpoint(book_slug)
        if cp is None:
            raise FileNotFoundError(f"No checkpoint found for '{book_slug}'")
        cp["phase"] = phase
        cp["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write(cp, book_slug)
        return cp

    def update_artifacts(self, book_slug: str, artifacts: dict) -> dict:
        """Set artifact paths in checkpoint."""
        cp = self.load_checkpoint(book_slug)
        if cp is None:
            raise FileNotFoundError(f"No checkpoint found for '{book_slug}'")
        cp["artifacts"].update(artifacts)
        cp["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write(cp, book_slug)
        return cp

    def add_error(self, book_slug: str, phase: str, message: str, recoverable: bool = True) -> None:
        """Log an error to checkpoint without failing the pipeline."""
        cp = self.load_checkpoint(book_slug)
        if cp is None:
            return
        cp["errors"].append({
            "phase": phase,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": recoverable,
        })
        cp["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write(cp, book_slug)

    def build_dashboard(self) -> str:
        """Build a human-readable status dashboard across all books."""
        work_dir = Path(self.work_dir)
        if not work_dir.exists():
            return "No books in progress."

        lines = []
        for item in sorted(work_dir.iterdir()):
            cp_path = item / ".checkpoint.json"
            if not cp_path.exists():
                continue
            with open(cp_path) as f:
                cp = json.load(f)

            errors = len(cp.get("errors", []))
            chapters_done = sum(
                1 for ch in cp.get("chapters", [])
                if ch.get("status") == "done"
            )
            chapters_total = len(cp.get("chapters", []))
            phase = cp.get("phase", "unknown")

            lines.append(
                f"{cp['book_slug']:30s} phase={phase:12s} "
                f"chapters={chapters_done}/{chapters_total} "
                f"errors={errors}"
            )

        if not lines:
            return "No books in progress."

        return "\n".join([
            "Lumen Status Dashboard",
            "=" * 60,
            *lines,
        ])

    def build_dashboard_json(self) -> dict[str, Any]:
        """Build machine-readable dashboard as dict with phase grouping."""
        work_dir = Path(self.work_dir)
        result: dict[str, Any] = {
            "books": [],
            "summary": {"total": 0, "by_phase": {}},
        }
        if not work_dir.exists():
            return result

        for item in sorted(work_dir.iterdir()):
            cp_path = item / ".checkpoint.json"
            if not cp_path.exists():
                continue
            with open(cp_path) as f:
                cp = json.load(f)

            errors = len(cp.get("errors", []))
            chapters = cp.get("chapters", [])
            chapters_done = sum(1 for ch in chapters if ch.get("status") == "done")
            chapters_total = len(chapters)
            phase = cp.get("phase", "unknown")

            result["books"].append({
                "slug": cp["book_slug"],
                "book_path": cp.get("book_path", ""),
                "content_type": cp.get("content_type", "unknown"),
                "phase": phase,
                "chapters_done": chapters_done,
                "chapters_total": chapters_total,
                "errors": errors,
                "updated_at": cp.get("updated_at", ""),
            })
            result["summary"]["total"] += 1
            result["summary"]["by_phase"][phase] = result["summary"]["by_phase"].get(phase, 0) + 1

        return result

    def _write(self, checkpoint: dict, book_slug: str) -> None:
        path = self._checkpoint_path(book_slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def resolve_slug(self, slug_or_path: str) -> str:
        """Resolve a book slug, auto-detecting from checkpoint if ambiguous.

        Results are cached per CheckpointManager instance (lazy init on first call).
        """
        from pathlib import Path

        # First try: direct checkpoint hit
        cp = self.load_checkpoint(slug_or_path)
        if cp is not None:
            return slug_or_path

        # Build slug cache lazily
        if self._slug_cache is None:
            self._slug_cache = {}
            work_dir = Path(self.work_dir)
            if work_dir.exists():
                for item in sorted(work_dir.iterdir()):
                    cp_path = item / ".checkpoint.json"
                    if not cp_path.exists():
                        continue
                    with open(cp_path) as f:
                        cp = json.load(f)
                    bp = cp.get("book_path", "")
                    if bp:
                        self._slug_cache[bp] = cp.get("book_slug", item.name)

        # Second try: look up by book_path in cache
        for bp, slug in self._slug_cache.items():
            if bp.endswith(slug_or_path) or slug_or_path in bp:
                print(f"[Checkpoint] Resolved slug '{slug_or_path}' -> '{slug}'")
                return slug

        return slug_or_path
