# ABOUTME: Tests for SQLite and Foundry data source backends including CRUD operations,
# ABOUTME: schema handling, and data persistence behavior.
"""Tests for SQLite and Foundry data source backends including CRUD operations,."""

import uuid

import pytest

from mdfactory.utils.data_manager import FoundryDataSource, SQLiteDataSource


def test_basic_sql(tmp_path):
    db_path = tmp_path / "test.db"
    ds = SQLiteDataSource(db_path, table_name="test", allow_null=False)

    exists = ds._table_exists()
    assert not exists
    print(ds._columns())

    ds.delete_data({})

    ds.save_data([{"a": 1, "b": 2}])
    ds.save_data([{"a": 1, "b": 2}])
    assert ds._table_exists()
    print(ds._columns())

    with pytest.raises(ValueError, match="names do not match"):
        ds.save_data([{"a": 1, "b": 2}, {"d": 7}])

    df = ds.load_data()
    assert len(df) == 1
    ds.save_data([{"a": 10, "b": 20}])
    df = ds.load_data()
    assert len(df) == 2
    print(df)

    ds.delete_data({"a": 1})
    df = ds.load_data()
    assert len(df) == 1
    assert df.iloc[0]["a"] == 10

    ds.delete_data({})
    assert ds.load_data().empty

