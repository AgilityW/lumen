import json
import os
import tempfile

import pytest

from lumen.core.state import CheckpointManager


@pytest.fixture
def work_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


class TestCheckpointManager:
    def test_init_checkpoint(self, work_dir):
        mgr = CheckpointManager(work_dir=work_dir)
        cp = mgr.init_checkpoint("test-book", "/path/to/book.epub", "epub", content_type="book")
        assert cp["book_slug"] == "test-book"
        assert cp["book_path"] == "/path/to/book.epub"
        assert cp["book_format"] == "epub"
        assert cp["content_type"] == "book"
        assert cp["phase"] == "init"
        assert cp["version"] == 1
        assert cp["created_at"]
        assert cp["updated_at"]

    def test_load_checkpoint(self, work_dir):
        mgr = CheckpointManager(work_dir=work_dir)
        mgr.init_checkpoint("test-book", "/p/b.epub", "epub")
        loaded = mgr.load_checkpoint("test-book")
        assert loaded is not None
        assert loaded["book_slug"] == "test-book"

    def test_load_nonexistent(self, work_dir):
        mgr = CheckpointManager(work_dir=work_dir)
        assert mgr.load_checkpoint("no-such-book") is None

    def test_update_phase(self, work_dir):
        mgr = CheckpointManager(work_dir=work_dir)
        mgr.init_checkpoint("test-book", "/p/b.epub", "epub")
        mgr.update_phase("test-book", "skeletonize")
        cp = mgr.load_checkpoint("test-book")
        assert cp["phase"] == "skeletonize"

    def test_update_phase_nonexistent_raises(self, work_dir):
        mgr = CheckpointManager(work_dir=work_dir)
        with pytest.raises(FileNotFoundError):
            mgr.update_phase("no-such-book", "skeletonize")

    def test_update_artifacts(self, work_dir):
        mgr = CheckpointManager(work_dir=work_dir)
        mgr.init_checkpoint("test-book", "/p/b.epub", "epub")
        mgr.update_artifacts("test-book", {"skeleton": "/tmp/skeleton.json"})
        cp = mgr.load_checkpoint("test-book")
        assert cp["artifacts"]["skeleton"] == "/tmp/skeleton.json"

    def test_add_error(self, work_dir):
        mgr = CheckpointManager(work_dir=work_dir)
        mgr.init_checkpoint("test-book", "/p/b.epub", "epub")
        mgr.add_error("test-book", "skeletonize", "Something went wrong")
        cp = mgr.load_checkpoint("test-book")
        assert len(cp["errors"]) == 1
        assert cp["errors"][0]["phase"] == "skeletonize"
        assert cp["errors"][0]["recoverable"] is True

    def test_add_error_nonexistent_no_raise(self, work_dir):
        mgr = CheckpointManager(work_dir=work_dir)
        mgr.add_error("no-such-book", "phase", "msg")

    def test_build_dashboard_no_books(self, work_dir):
        mgr = CheckpointManager(work_dir=work_dir)
        assert "No books in progress" in mgr.build_dashboard()

    def test_build_dashboard_with_books(self, work_dir):
        mgr = CheckpointManager(work_dir=work_dir)
        mgr.init_checkpoint("book-1", "/p/1.epub", "epub")
        mgr.update_phase("book-1", "skeletonize")
        mgr.init_checkpoint("book-2", "/p/2.epub", "epub")
        mgr.update_phase("book-2", "complete")
        dash = mgr.build_dashboard()
        assert "book-1" in dash
        assert "book-2" in dash

    def test_build_dashboard_json(self, work_dir):
        mgr = CheckpointManager(work_dir=work_dir)
        mgr.init_checkpoint("book-1", "/p/1.epub", "epub")
        mgr.update_phase("book-1", "digest")
        data = mgr.build_dashboard_json()
        assert data["summary"]["total"] == 1
        assert data["summary"]["by_phase"].get("digest", 0) == 1
        assert data["books"][0]["slug"] == "book-1"

    def test_resolve_slug_direct_hit(self, work_dir):
        mgr = CheckpointManager(work_dir=work_dir)
        mgr.init_checkpoint("my-book", "/p/b.epub", "epub")
        assert mgr.resolve_slug("my-book") == "my-book"

    def test_resolve_slug_fallback(self, work_dir):
        mgr = CheckpointManager(work_dir=work_dir)
        assert mgr.resolve_slug("new-slug") == "new-slug"

    def test_persistence_after_reinit(self, work_dir):
        mgr1 = CheckpointManager(work_dir=work_dir)
        mgr1.init_checkpoint("persist-test", "/p/b.epub", "epub")
        mgr1.update_phase("persist-test", "deep-read")

        mgr2 = CheckpointManager(work_dir=work_dir)
        cp = mgr2.load_checkpoint("persist-test")
        assert cp["phase"] == "deep-read"

    def test_default_work_dir(self):
        mgr = CheckpointManager()
        assert mgr.work_dir is not None
