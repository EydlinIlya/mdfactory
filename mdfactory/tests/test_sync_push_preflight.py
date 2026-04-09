# ABOUTME: Tests for CLI preflight checks before push operations
# ABOUTME: Validates _ensure_sync_target_initialized for different backends

import sys

import pytest

from mdfactory import cli

# Get the actual settings module (not the singleton attribute)
_settings_mod = sys.modules["mdfactory.settings"]


def test_ensure_sync_target_initialized_skips_foundry(monkeypatch):
    class FakeSettings:
        def __init__(self):
            self.config = {"database": {"TYPE": "foundry"}}

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("DataManager should not be called for foundry backend")

    monkeypatch.setattr(_settings_mod, "Settings", FakeSettings)
    monkeypatch.setattr("mdfactory.utils.data_manager.DataManager", fail_if_called)

    cli._ensure_sync_target_initialized("ANALYSIS_OVERVIEW", "mdfactory sync init analysis")


def test_ensure_sync_target_initialized_exits_when_missing(monkeypatch):
    class FakeSettings:
        def __init__(self):
            self.config = {"database": {"TYPE": "sqlite"}}

    class FakeDataManager:
        def __init__(self, _table_name: str):
            raise FileNotFoundError("SQLite database file does not exist")

    monkeypatch.setattr(_settings_mod, "Settings", FakeSettings)
    monkeypatch.setattr("mdfactory.utils.data_manager.DataManager", FakeDataManager)

    with pytest.raises(SystemExit) as exc:
        cli._ensure_sync_target_initialized("ANALYSIS_OVERVIEW", "mdfactory sync init analysis")
    assert exc.value.code == 1
