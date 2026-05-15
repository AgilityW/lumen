"""Shared config loading — single source of truth for all modules."""

import os
from typing import Any

import yaml

_CONFIG_PATHS = [
    "config.yaml",
    os.path.expanduser("~/.lumen/config.yaml"),
]

_config_cache: dict | None = None


def find_config() -> str | None:
    """Locate config.yaml on disk."""
    for path in _CONFIG_PATHS:
        if os.path.exists(path):
            return path
    # Fallback: look relative to this file's project root
    _base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    project_config = os.path.join(_base, "config.yaml")
    if os.path.exists(project_config):
        return project_config
    return None


def load_config() -> dict[str, Any]:
    """Load config from disk with module-level caching (singleton per process)."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    config_path = find_config()
    if config_path:
        with open(config_path) as f:
            _config_cache = yaml.safe_load(f) or {}
    else:
        _config_cache = {}
    return _config_cache


def invalidate_config_cache() -> None:
    """Clear cached config (used by init_config to force reload)."""
    global _config_cache
    _config_cache = None


def default_config() -> dict[str, Any]:
    """Return the default config dict (suitable for init)."""
    return {
        "api": {
            "backend": "deepseek",
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
            },
            "claude": {
                "base_url": "https://api.anthropic.com",
                "model": "claude-sonnet-4-20250514",
            },
        },
        "vault": {
            "path": "",
            "book_dir": "Books",
        },
        "output": {
            "work_dir": "output",
        },
        "framework": {
            "default": "technical",
            "path": "frameworks",
        },
        "checkpoint": {
            "enabled": True,
            "auto_save": True,
        },
    }
