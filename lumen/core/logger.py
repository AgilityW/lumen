"""Lightweight logging setup for Lumen.

Replaces bare print() with structured logging. Call setup_logging() once
at CLI entry point to configure output format and verbosity.
"""

import logging
import sys
from typing import Any


_LOG = logging.getLogger("lumen")
_LOG_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warn": logging.WARNING, "error": logging.ERROR}


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a child logger scoped to the given name, e.g. get_logger("pipeline")."""
    return _LOG.getChild(name) if name else _LOG


def setup_logging(level: str = "info") -> None:
    """Configure the root lumen logger with a consistent format.

    Call once at startup (CLI entry point or run.sh).
    """
    fmt = logging.Formatter(
        fmt="%(message)s",
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    _LOG.addHandler(handler)
    _LOG.setLevel(_LOG_LEVELS.get(level, logging.INFO))


def phase(msg: str, phase_num: int = 1) -> None:
    """Print a Phase header, e.g. phase("Ingestion", 1)."""
    _LOG.info("[Phase %d] %s", phase_num, msg)


def ok(msg: str) -> None:
    _LOG.info("[OK] %s", msg)


def warn(msg: str) -> None:
    _LOG.warning("[WARN] %s", msg)


def err(msg: str) -> None:
    _LOG.error("[ERROR] %s", msg)
