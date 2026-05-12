import os
import tempfile

import pytest
import yaml

from lumen.core.config import find_config, load_config, default_config, invalidate_config_cache


class TestConfig:
    def test_default_config_structure(self):
        cfg = default_config()
        assert "api" in cfg
        assert "vault" in cfg
        assert "output" in cfg
        assert "framework" in cfg
        assert "checkpoint" in cfg
        assert cfg["api"]["backend"] == "deepseek"
        assert cfg["api"]["deepseek"]["api_key"] == ""

    def test_load_config_cache_invalidation(self):
        invalidate_config_cache()
        cfg1 = load_config()
        cfg2 = load_config()
        assert cfg1 is cfg2
        invalidate_config_cache()
        cfg3 = load_config()
        assert cfg1 is not cfg3

    def test_find_config_falls_back_to_project_root(self):
        # find_config checks the project root relative to the module file,
        # so it can find config.yaml even when CWD has no config.
        found = find_config()
        assert found is not None
        assert found.endswith("config.yaml")

    def test_config_yaml_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = default_config()
            cfg_path = os.path.join(tmp, "config.yaml")
            with open(cfg_path, "w") as f:
                yaml.dump(cfg, f)

            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                invalidate_config_cache()
                loaded = load_config()
                assert loaded["api"]["backend"] == "deepseek"
            finally:
                os.chdir(old_cwd)

    def test_key_fallback_chain(self):
        cfg = default_config()
        assert cfg["api"]["deepseek"]["api_key"] == ""
        assert cfg["api"]["claude"]["api_key"] == ""

    def test_vault_defaults(self):
        cfg = default_config()
        assert cfg["vault"]["path"] == ""
        assert cfg["vault"]["book_dir"] == "Books"
