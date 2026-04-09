# ABOUTME: Tests for database operations including record upload, local upload modes,
# ABOUTME: and DataManager interactions with SQLite storage.
"""Tests for database operations including record upload, local upload modes,."""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone

import pandas as pd
import pytest

from mdfactory.utils.data_manager import DataManager, quote_sqlite_identifier
from mdfactory.utils.db_operations import (
    local_upload_with_modes,
    upload_records,
)

from .conftest import make_test_record


def _overview_record(
    hash_value: str,
    item_name: str,
    status: str = "completed",
) -> dict[str, object]:
    return {
        "hash": hash_value,
        "simulation_type": "bilayer",
        "directory": f"/tmp/{hash_value}",
        "item_type": "analysis",
        "item_name": item_name,
        "status": status,
        "row_count": 1 if status == "completed" else 0,
        "file_count": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def test_local_upload_with_modes_default_insert(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    records = [make_test_record("HASH1"), make_test_record("HASH2")]

    count = local_upload_with_modes(
        dm=dm,
        records=records,
        key_fields=["hash"],
        table_name="RUN_DATABASE",
        force=False,
        diff=False,
    )

    assert count == 2
    assert len(dm.load_data()) == 2


def test_local_upload_with_modes_duplicate_error(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    dm.save_data(make_test_record("EXISTING"))

    with pytest.raises(ValueError, match="already exist"):
        local_upload_with_modes(
            dm=dm,
            records=[make_test_record("EXISTING")],
            key_fields=["hash"],
            table_name="RUN_DATABASE",
            force=False,
            diff=False,
        )


def test_local_upload_with_modes_diff_skips_existing(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    dm.save_data(make_test_record("EXISTING"))

    count = local_upload_with_modes(
        dm=dm,
        records=[make_test_record("EXISTING"), make_test_record("NEW")],
        key_fields=["hash"],
        table_name="RUN_DATABASE",
        force=False,
        diff=True,
    )

    assert count == 1
    assert set(dm.load_data()["hash"]) == {"EXISTING", "NEW"}


def test_local_upload_with_modes_force_overwrites(temp_run_db):
    dm = DataManager("RUN_DATABASE")
    dm.delete_data({})
    dm.save_data(make_test_record("HASH1", status="build"))

    count = local_upload_with_modes(
        dm=dm,
        records=[make_test_record("HASH1", status="completed")],
        key_fields=["hash"],
        table_name="RUN_DATABASE",
        force=True,
        diff=False,
    )

    assert count == 1
    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["status"] == "completed"


def test_upload_records_force_and_diff_rejected(temp_run_db):
    """Test upload_records rejects force=True and diff=True together."""
    records = [make_test_record("HASH1")]

    with pytest.raises(ValueError, match="force and diff"):
        upload_records(records, "RUN_DATABASE", ["hash"], force=True, diff=True)


def test_local_upload_with_modes_composite_key_force(temp_analysis_db):
    dm = DataManager("ANALYSIS_OVERVIEW")
    dm.delete_data({})
    dm.save_data(_overview_record("HASH_A", "area_per_lipid", status="not_yet_run"))

    count = local_upload_with_modes(
        dm=dm,
        records=[_overview_record("HASH_A", "area_per_lipid", status="completed")],
        key_fields=["hash", "item_type", "item_name"],
        table_name="ANALYSIS_OVERVIEW",
        force=True,
        diff=False,
    )

    assert count == 1
    df = dm.load_data()
    assert len(df) == 1
    assert df.iloc[0]["status"] == "completed"


def _install_fake_foundry_module(monkeypatch, dataset):
    class _FakeFoundryContext:
        def get_dataset_by_path(
            self,
            path: str,
            create_if_not_exist: bool = False,
            create_branch_if_not_exists: bool = False,
        ):
            return dataset

    monkeypatch.setitem(
        sys.modules,
        "foundry_dev_tools",
        types.SimpleNamespace(FoundryContext=_FakeFoundryContext),
    )


def test_init_foundry_tables_mismatch_raises_without_reset(monkeypatch):
    from mdfactory.utils import db_operations

    class _FakeDataset:
        rid = "ri.fake.dataset"

        def __init__(self):
            self.transaction = {}

        def query_foundry_sql(self, _query: str):
            return pd.DataFrame(columns=["hash", "unexpected"])

    class _FakeDataManager:
        def __init__(self, _table_name: str):
            pass

        def delete_data(self, _conditions: dict):
            return None

    dataset = _FakeDataset()
    _install_fake_foundry_module(monkeypatch, dataset)
    monkeypatch.setattr(db_operations.FoundryDataSource, "dataset_exists", lambda _path: True)
    monkeypatch.setattr(db_operations, "DataManager", _FakeDataManager)

    with pytest.raises(ValueError, match="schema mismatch"):
        db_operations.init_foundry_tables(
            [("RUN_DATABASE", {"hash": "__PLACEHOLDER__"}, ["hash", "engine"])],
            reset=False,
        )


def test_init_foundry_tables_reset_recreates_on_schema_match(monkeypatch):
    from mdfactory.utils import db_operations

    class _FakeDataset:
        rid = "ri.fake.dataset"

        def __init__(self):
            self.transaction = {}
            self.upload_calls = 0

        def query_foundry_sql(self, _query: str):
            return pd.DataFrame(columns=["hash", "engine"])

        def upload_file(self, **_kwargs):
            self.upload_calls += 1

        def infer_schema(self):
            return {"fields": []}

        def start_transaction(self, start_transaction_type: str = "UPDATE"):
            self.transaction = {"rid": f"ri.transaction.{start_transaction_type.lower()}"}

        def upload_schema(self, _transaction_rid: str, _schema):
            return None

        def commit_transaction(self):
            return None

    class _FakeDataManager:
        def __init__(self, _table_name: str):
            pass

        def delete_data(self, _conditions: dict):
            return None

    dataset = _FakeDataset()
    _install_fake_foundry_module(monkeypatch, dataset)
    monkeypatch.setattr(db_operations.FoundryDataSource, "dataset_exists", lambda _path: True)
    monkeypatch.setattr(db_operations, "DataManager", _FakeDataManager)
    monkeypatch.setattr(db_operations, "wait_for_schema", lambda _dataset, _columns: None)

    results = db_operations.init_foundry_tables(
        [
            (
                "RUN_DATABASE",
                {"hash": "__PLACEHOLDER__", "engine": "gromacs"},
                ["hash", "engine"],
            )
        ],
        reset=True,
    )

    assert results["RUN_DATABASE"] is True
    assert dataset.upload_calls == 1


def test_init_foundry_tables_existing_without_reset_noop(monkeypatch):
    from mdfactory.utils import db_operations

    class _FakeDataset:
        rid = "ri.fake.dataset"

        def __init__(self):
            self.transaction = {}
            self.upload_calls = 0

        def query_foundry_sql(self, _query: str):
            return pd.DataFrame(columns=["hash", "engine"])

        def upload_file(self, **_kwargs):
            self.upload_calls += 1

    dataset = _FakeDataset()
    _install_fake_foundry_module(monkeypatch, dataset)
    monkeypatch.setattr(db_operations.FoundryDataSource, "dataset_exists", lambda _path: True)

    results = db_operations.init_foundry_tables(
        [
            (
                "RUN_DATABASE",
                {"hash": "__PLACEHOLDER__", "engine": "gromacs"},
                ["hash", "engine"],
            )
        ],
        reset=False,
    )

    assert results["RUN_DATABASE"] is False
    assert dataset.upload_calls == 0


def test_quote_sqlite_identifier_rejects_invalid():
    with pytest.raises(ValueError, match="Invalid SQLite identifier"):
        quote_sqlite_identifier("RUN_DATABASE; DROP TABLE RUN_DATABASE;")

    assert quote_sqlite_identifier("RUN_DATABASE") == '"RUN_DATABASE"'


def test_init_foundry_tables_fails_fast_on_auth(monkeypatch):
    from mdfactory.utils import db_operations

    calls = {"dataset_calls": 0}

    class _FakeMultipass:
        @staticmethod
        def get_user_info():
            raise RuntimeError("auth failed")

    class _FakeFoundryContext:
        def __init__(self):
            self.multipass = _FakeMultipass()

        def get_dataset_by_path(
            self,
            path: str,
            create_if_not_exist: bool = False,
            create_branch_if_not_exists: bool = False,
        ):
            calls["dataset_calls"] += 1
            raise AssertionError(f"Unexpected dataset call for {path}")

    monkeypatch.setitem(
        sys.modules,
        "foundry_dev_tools",
        types.SimpleNamespace(FoundryContext=_FakeFoundryContext),
    )

    with pytest.raises(ValueError, match="connectivity/authentication"):
        db_operations.init_foundry_tables(
            [("RUN_DATABASE", {"hash": "__PLACEHOLDER__"}, ["hash"])],
            reset=False,
        )

    assert calls["dataset_calls"] == 0
