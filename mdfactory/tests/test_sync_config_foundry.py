# ABOUTME: Tests for Foundry backend configuration in the config wizard
# ABOUTME: Validates fdt checks, path setup, and sync init check command

from __future__ import annotations

import configparser
import sys
import types

import pytest

from mdfactory.cli import sync_init_check
from mdfactory.settings import Settings
from mdfactory.utils.sync_config import configure_foundry_paths


class _FakeResponse:
    def __init__(self, status_code: int, rid: str = "ri.fake.dataset"):
        self.status_code = status_code
        self._rid = rid

    def json(self) -> dict[str, str]:
        return {"rid": self._rid}


def _install_fake_foundry_module(
    monkeypatch,
    *,
    status_by_path: dict[str, int] | None = None,
    default_status: int = 200,
) -> None:
    status_by_path = status_by_path or {}

    class _FakeCompass:
        def api_get_resource_by_path(self, path: str) -> _FakeResponse:
            status = status_by_path.get(path, default_status)
            return _FakeResponse(status_code=status)

    class _FakeMultipass:
        @staticmethod
        def get_user_info() -> dict[str, str]:
            return {"username": "test-user"}

    class _FakeFoundryContext:
        def __init__(self):
            self.compass = _FakeCompass()
            self.multipass = _FakeMultipass()

    monkeypatch.setitem(
        sys.modules,
        "foundry_dev_tools",
        types.SimpleNamespace(FoundryContext=_FakeFoundryContext),
    )


def _base_foundry_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "foundry": {
                "BASE_PATH": "/Group Functions/mdfactory",
                "ANALYSIS_NAME": "analysis",
                "ANALYSIS_DB_PATH": "/Group Functions/mdfactory/analysis",
                "ARTIFACT_DB_PATH": "/Group Functions/mdfactory/artifacts",
                "RUN_DB_PATH": "/Group Functions/mdfactory/runs",
            }
        }
    )
    return config


class _FakeQuestion:
    def __init__(self, value):
        self._value = value

    def ask(self):
        return self._value


def test_configure_foundry_paths_missing_fdt(monkeypatch):
    config = _base_foundry_config()
    monkeypatch.setattr("mdfactory.utils.sync_config.shutil.which", lambda _: None)

    with pytest.raises(SystemExit):
        configure_foundry_paths(config)


def test_configure_foundry_paths_fdt_config_failure(monkeypatch):
    config = _base_foundry_config()
    monkeypatch.setattr("mdfactory.utils.sync_config.shutil.which", lambda _: "/usr/bin/fdt")
    monkeypatch.setattr(
        "mdfactory.utils.sync_config.subprocess.run",
        lambda *_, **__: types.SimpleNamespace(returncode=1),
    )

    with pytest.raises(SystemExit):
        configure_foundry_paths(config)


def test_configure_foundry_paths_success_sets_expected_paths(monkeypatch):
    config = _base_foundry_config()
    _install_fake_foundry_module(monkeypatch)

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool = False):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0)

    # Mock questionary: base_path (default), analysis_name (default), confirm use paths (True)
    text_answers = iter(["/Group Functions/mdfactory", "analysis"])
    confirm_answers = iter([True])

    monkeypatch.setattr(
        "mdfactory.utils.sync_config.questionary.text",
        lambda *a, **kw: _FakeQuestion(next(text_answers)),
    )
    monkeypatch.setattr(
        "mdfactory.utils.sync_config.questionary.confirm",
        lambda *a, **kw: _FakeQuestion(next(confirm_answers)),
    )
    monkeypatch.setattr("mdfactory.utils.sync_config.shutil.which", lambda _: "/usr/bin/fdt")
    monkeypatch.setattr("mdfactory.utils.sync_config.subprocess.run", fake_run)

    configure_foundry_paths(config)

    assert calls == [["fdt", "config"]]
    assert config["foundry"]["BASE_PATH"] == "/Group Functions/mdfactory"
    assert config["foundry"]["ANALYSIS_NAME"] == "analysis"
    assert config["foundry"]["ANALYSIS_DB_PATH"] == "/Group Functions/mdfactory/analysis"
    assert config["foundry"]["ARTIFACT_DB_PATH"] == "/Group Functions/mdfactory/artifacts"
    assert config["foundry"]["RUN_DB_PATH"] == "/Group Functions/mdfactory/runs"


def test_configure_foundry_paths_rejects_invalid_analysis_name(monkeypatch):
    config = _base_foundry_config()
    _install_fake_foundry_module(monkeypatch)

    text_answers = iter(["/Group Functions/mdfactory", "bad/name"])

    monkeypatch.setattr(
        "mdfactory.utils.sync_config.questionary.text",
        lambda *a, **kw: _FakeQuestion(next(text_answers)),
    )
    monkeypatch.setattr("mdfactory.utils.sync_config.shutil.which", lambda _: "/usr/bin/fdt")
    monkeypatch.setattr(
        "mdfactory.utils.sync_config.subprocess.run",
        lambda *_, **__: types.SimpleNamespace(returncode=0),
    )

    with pytest.raises(SystemExit):
        configure_foundry_paths(config)


def test_sync_init_check_foundry_success(monkeypatch):
    paths = {
        "BASE_PATH": "/Group Functions/mdfactory-ci/test",
        "ANALYSIS_DB_PATH": "/Group Functions/mdfactory-ci/test/analysis",
        "ARTIFACT_DB_PATH": "/Group Functions/mdfactory-ci/test/artifacts",
        "RUN_DB_PATH": "/Group Functions/mdfactory-ci/test/runs",
    }
    _install_fake_foundry_module(
        monkeypatch,
        status_by_path={path: 200 for path in paths.values()},
    )

    def patched_load_config(self):
        self.config = configparser.ConfigParser()
        self.config.read_dict({"database": {"TYPE": "foundry"}, "foundry": paths})

    monkeypatch.setattr(Settings, "_load_config", patched_load_config)
    sync_init_check()


def test_sync_init_check_foundry_path_failure(monkeypatch):
    paths = {
        "BASE_PATH": "/Group Functions/mdfactory-ci/test",
        "ANALYSIS_DB_PATH": "/Group Functions/mdfactory-ci/test/analysis",
        "ARTIFACT_DB_PATH": "/Group Functions/mdfactory-ci/test/artifacts",
        "RUN_DB_PATH": "/Group Functions/mdfactory-ci/test/runs",
    }
    status_by_path = {path: 200 for path in paths.values()}
    status_by_path[paths["ARTIFACT_DB_PATH"]] = 404
    _install_fake_foundry_module(monkeypatch, status_by_path=status_by_path)

    def patched_load_config(self):
        self.config = configparser.ConfigParser()
        self.config.read_dict({"database": {"TYPE": "foundry"}, "foundry": paths})

    monkeypatch.setattr(Settings, "_load_config", patched_load_config)
    with pytest.raises(SystemExit):
        sync_init_check()
