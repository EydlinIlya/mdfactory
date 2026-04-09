# ABOUTME: Live Foundry integration tests requiring real Foundry credentials
# ABOUTME: Opt-in via MDF_TEST_FOUNDRY=1 and MDF_TEST_FOUNDRY_BASE_PATH env vars

from __future__ import annotations

import os

import pytest

try:
    from foundry_dev_tools import FoundryContext
except ImportError:  # pragma: no cover - optional dependency for live tests
    FoundryContext = None

pytestmark = pytest.mark.foundry_live


def _require_foundry_env() -> str:
    if os.getenv("MDF_TEST_FOUNDRY") != "1":
        pytest.skip("Set MDF_TEST_FOUNDRY=1 to run live Foundry tests.")

    base_path = os.getenv("MDF_TEST_FOUNDRY_BASE_PATH")
    if not base_path:
        pytest.skip("Set MDF_TEST_FOUNDRY_BASE_PATH to run live Foundry tests.")
    if not base_path.startswith("/"):
        pytest.skip("MDF_TEST_FOUNDRY_BASE_PATH must be an absolute Foundry path.")
    return base_path


def test_foundry_live_env_vars_present():
    base_path = _require_foundry_env()
    assert base_path.startswith("/")


def test_foundry_live_base_path_accessible():
    base_path = _require_foundry_env()
    if FoundryContext is None:
        pytest.skip("foundry_dev_tools is not installed.")

    ctx = FoundryContext()
    user_info = ctx.multipass.get_user_info()
    assert user_info

    response = ctx.compass.api_get_resource_by_path(base_path)
    assert response.status_code == 200
