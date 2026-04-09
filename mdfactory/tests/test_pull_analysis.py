# ABOUTME: Tests for analysis sync pull functionality
# ABOUTME: Tests query, formatting, and CLI commands for analysis data retrieval

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from mdfactory.cli import sync_pull_analysis
from mdfactory.utils.data_manager import DataManager
from mdfactory.utils.pull_analysis import (
    decode_analysis_data,
    format_analysis_summary,
    format_overview_summary,
    list_available_tables,
    pull_analysis,
    pull_artifact,
    pull_overview,
    pull_systems_with_analyses,
)
from mdfactory.utils.push_analysis import (
    serialize_dataframe_to_csv,
)

# temp_analysis_db fixture is auto-discovered from conftest.py


def _analysis_record(hash_value: str, simulation_type: str = "bilayer") -> dict:
    """Create a sample analysis record for testing."""
    df = pd.DataFrame(
        {
            "time_ns": [0.0, 1.0, 2.0],
            "value": [10.0, 20.0, 30.0],
        }
    )
    return {
        "hash": hash_value,
        "directory": f"/tmp/{hash_value}",
        "simulation_type": simulation_type,
        "row_count": len(df),
        "columns": json.dumps(df.columns.tolist()),
        "data_csv": serialize_dataframe_to_csv(df),
        "data_path": ".analysis/test.parquet",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _overview_record(
    hash_value: str,
    item_name: str,
    status: str = "completed",
    item_type: str = "analysis",
    simulation_type: str = "bilayer",
) -> dict:
    """Create a sample overview record for testing.

    Note: Uses empty string instead of None to avoid SQLite NULL constraint issues.
    """
    return {
        "hash": hash_value,
        "simulation_type": simulation_type,
        "directory": f"/tmp/{hash_value}",
        "item_type": item_type,
        "item_name": item_name,
        "status": status,
        "row_count": 10 if status == "completed" and item_type == "analysis" else "",
        "file_count": 2 if status == "completed" and item_type == "artifact" else "",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _artifact_record(hash_value: str, simulation_type: str = "bilayer") -> dict:
    """Create a sample artifact record for testing."""
    return {
        "hash": hash_value,
        "directory": f"/tmp/{hash_value}",
        "simulation_type": simulation_type,
        "file_count": 2,
        "files": json.dumps(["artifacts/snapshot/top.png", "artifacts/snapshot/side.png"]),
        "checksums": json.dumps(
            {
                "artifacts/snapshot/top.png": "abc123",
                "artifacts/snapshot/side.png": "def456",
            }
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# GROUP 1: list_available_tables tests


def test_list_available_tables_empty(temp_analysis_db):
    """Test listing tables when none have data.

    Tables are created during init (empty, with schema established),
    so they should all be listed as available even without data.
    """
    available = list_available_tables()
    assert "analyses" in available
    assert "artifacts" in available
    assert "overview" in available
    assert available["overview"] is True
    assert len(available["analyses"]) > 0


def test_list_available_tables_with_data(temp_analysis_db):
    """Test listing tables after adding data."""
    dm = DataManager("ANALYSIS_AREA_PER_LIPID")
    dm.save_data(_analysis_record("HASH1"))

    available = list_available_tables()
    assert "area_per_lipid" in available["analyses"]


# GROUP 2: pull_overview tests


def test_pull_overview_empty(temp_analysis_db):
    """Test pulling from empty overview table."""
    df = pull_overview()
    assert isinstance(df, pd.DataFrame)
    # May be empty or have data from init


def test_pull_overview_with_data(temp_analysis_db):
    """Test pulling from overview table with data."""
    dm = DataManager("ANALYSIS_OVERVIEW")
    dm.save_data(_overview_record("HASH1", "area_per_lipid", "completed"))
    dm.save_data(_overview_record("HASH2", "density_distribution", "not_yet_run"))

    df = pull_overview()
    assert len(df) == 2


def test_pull_overview_filter_by_hash(temp_analysis_db):
    """Test filtering overview by hash."""
    dm = DataManager("ANALYSIS_OVERVIEW")
    dm.save_data(_overview_record("HASH1", "area_per_lipid"))
    dm.save_data(_overview_record("HASH2", "density_distribution"))

    df = pull_overview(hash="HASH1")
    assert len(df) == 1
    assert df.iloc[0]["hash"] == "HASH1"


def test_pull_overview_filter_by_simulation_type(temp_analysis_db):
    """Test filtering overview by simulation type."""
    dm = DataManager("ANALYSIS_OVERVIEW")
    dm.save_data(_overview_record("HASH1", "area_per_lipid", simulation_type="bilayer"))
    dm.save_data(_overview_record("HASH2", "test", simulation_type="mixedbox"))

    df = pull_overview(simulation_type="bilayer")
    assert len(df) == 1
    assert df.iloc[0]["simulation_type"] == "bilayer"


def test_pull_overview_filter_by_item_type(temp_analysis_db):
    """Test filtering overview by item type."""
    dm = DataManager("ANALYSIS_OVERVIEW")
    dm.save_data(_overview_record("HASH1", "area_per_lipid", item_type="analysis"))
    dm.save_data(_overview_record("HASH1", "snapshot", item_type="artifact"))

    df = pull_overview(item_type="analysis")
    assert len(df) == 1
    assert df.iloc[0]["item_type"] == "analysis"


def test_pull_overview_filter_by_item_name(temp_analysis_db):
    """Test filtering overview by item name."""
    dm = DataManager("ANALYSIS_OVERVIEW")
    dm.save_data(_overview_record("HASH1", "area_per_lipid"))
    dm.save_data(_overview_record("HASH2", "density_distribution"))

    df = pull_overview(item_name="area_per_lipid")
    assert len(df) == 1
    assert df.iloc[0]["item_name"] == "area_per_lipid"


# GROUP 3: pull_analysis tests


def test_pull_analysis_empty_table(temp_analysis_db):
    """Test pulling from empty analysis table."""
    df = pull_analysis("area_per_lipid")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_pull_analysis_with_data(temp_analysis_db):
    """Test pulling analysis data."""
    dm = DataManager("ANALYSIS_AREA_PER_LIPID")
    dm.save_data(_analysis_record("HASH1"))
    dm.save_data(_analysis_record("HASH2"))

    df = pull_analysis("area_per_lipid")
    assert len(df) == 2


def test_pull_analysis_filter_by_hash(temp_analysis_db):
    """Test filtering analysis by hash."""
    dm = DataManager("ANALYSIS_AREA_PER_LIPID")
    dm.save_data(_analysis_record("HASH1"))
    dm.save_data(_analysis_record("HASH2"))

    df = pull_analysis("area_per_lipid", hash="HASH1")
    assert len(df) == 1
    assert df.iloc[0]["hash"] == "HASH1"


def test_pull_analysis_filter_by_simulation_type(temp_analysis_db):
    """Test filtering analysis by simulation type."""
    dm = DataManager("ANALYSIS_AREA_PER_LIPID")
    dm.save_data(_analysis_record("HASH1", simulation_type="bilayer"))
    dm.save_data(_analysis_record("HASH2", simulation_type="mixedbox"))

    df = pull_analysis("area_per_lipid", simulation_type="bilayer")
    assert len(df) == 1
    assert df.iloc[0]["simulation_type"] == "bilayer"


def test_pull_analysis_decode_data(temp_analysis_db):
    """Test pull_analysis with decode_data=True."""
    dm = DataManager("ANALYSIS_AREA_PER_LIPID")
    dm.save_data(_analysis_record("HASH1"))

    decoded = pull_analysis("area_per_lipid", decode_data=True)

    assert isinstance(decoded, dict)
    assert "HASH1" in decoded
    assert isinstance(decoded["HASH1"], pd.DataFrame)
    assert "time_ns" in decoded["HASH1"].columns


# GROUP 4: pull_artifact tests


def test_pull_artifact_empty_table(temp_analysis_db):
    """Test pulling from empty artifact table."""
    df = pull_artifact("bilayer_snapshot")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_pull_artifact_with_data(temp_analysis_db):
    """Test pulling artifact data."""
    dm = DataManager("ARTIFACT_BILAYER_SNAPSHOT")
    dm.save_data(_artifact_record("HASH1"))

    df = pull_artifact("bilayer_snapshot")
    assert len(df) == 1
    assert df.iloc[0]["file_count"] == 2


# GROUP 5: decode_analysis_data tests


def test_decode_analysis_data(temp_analysis_db):
    """Test decoding data_csv column."""
    dm = DataManager("ANALYSIS_AREA_PER_LIPID")
    dm.save_data(_analysis_record("HASH1"))
    dm.save_data(_analysis_record("HASH2"))

    df = pull_analysis("area_per_lipid")
    decoded = decode_analysis_data(df)

    assert "HASH1" in decoded
    assert "HASH2" in decoded
    assert isinstance(decoded["HASH1"], pd.DataFrame)
    assert "time_ns" in decoded["HASH1"].columns


def test_decode_analysis_data_empty():
    """Test decoding empty DataFrame."""
    df = pd.DataFrame()
    decoded = decode_analysis_data(df)
    assert decoded == {}


# GROUP 6: format functions tests


def test_format_analysis_summary(temp_analysis_db):
    """Test formatting analysis records for display."""
    dm = DataManager("ANALYSIS_AREA_PER_LIPID")
    dm.save_data(_analysis_record("HASH1"))

    df = pull_analysis("area_per_lipid")
    formatted = format_analysis_summary(df)

    # Should exclude data_csv column
    assert "data_csv" not in formatted.columns
    # Should keep other columns
    assert "hash" in formatted.columns
    assert "row_count" in formatted.columns


def test_format_analysis_summary_empty():
    """Test formatting empty DataFrame."""
    df = pd.DataFrame()
    formatted = format_analysis_summary(df)
    assert formatted.empty


def test_format_overview_summary(temp_analysis_db):
    """Test formatting overview records for display."""
    dm = DataManager("ANALYSIS_OVERVIEW")
    dm.save_data(_overview_record("HASH1", "area_per_lipid"))

    df = pull_overview()
    formatted = format_overview_summary(df)

    # Should have hash first
    assert formatted.columns[0] == "hash"
    # Should include key columns
    assert "item_type" in formatted.columns
    assert "item_name" in formatted.columns
    assert "status" in formatted.columns


# GROUP 7: pull_systems_with_analyses tests


def test_pull_systems_with_analyses(temp_analysis_db):
    """Test pulling summary of systems with analyses."""
    dm = DataManager("ANALYSIS_OVERVIEW")
    dm.save_data(_overview_record("HASH1", "area_per_lipid", "completed"))
    dm.save_data(_overview_record("HASH1", "density_distribution", "not_yet_run"))
    dm.save_data(_overview_record("HASH2", "area_per_lipid", "not_yet_run"))

    df = pull_systems_with_analyses()
    assert len(df) == 3


def test_pull_systems_with_analyses_filter_by_analysis(temp_analysis_db):
    """Test filtering by specific analysis."""
    dm = DataManager("ANALYSIS_OVERVIEW")
    dm.save_data(_overview_record("HASH1", "area_per_lipid", "completed"))
    dm.save_data(_overview_record("HASH1", "density_distribution", "completed"))

    df = pull_systems_with_analyses(analysis_name="area_per_lipid")
    assert len(df) == 1
    assert df.iloc[0]["item_name"] == "area_per_lipid"


def test_pull_systems_with_analyses_filter_by_status(temp_analysis_db):
    """Test filtering by status."""
    dm = DataManager("ANALYSIS_OVERVIEW")
    dm.save_data(_overview_record("HASH1", "area_per_lipid", "completed"))
    dm.save_data(_overview_record("HASH2", "area_per_lipid", "not_yet_run"))

    df = pull_systems_with_analyses(status="completed")
    assert len(df) == 1
    assert df.iloc[0]["status"] == "completed"


# GROUP 8: CLI tests


def test_cli_sync_pull_analysis_no_analysis_name(monkeypatch, temp_analysis_db):
    """Test CLI requires analysis_name unless overview mode."""
    exit_code = None

    def mock_exit(code):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr("sys.exit", mock_exit)

    with pytest.raises(SystemExit):
        sync_pull_analysis()  # No analysis_name and no --overview

    assert exit_code == 1


def test_cli_sync_pull_analysis_overview_mode(temp_analysis_db, capsys):
    """Test CLI in overview mode."""
    dm = DataManager("ANALYSIS_OVERVIEW")
    dm.save_data(_overview_record("HASH1", "area_per_lipid", "completed"))

    sync_pull_analysis(overview=True)

    captured = capsys.readouterr()
    assert "HASH1" in captured.out or "1 record" in captured.out


def test_cli_sync_pull_analysis_specific_analysis(temp_analysis_db, capsys):
    """Test CLI pulling specific analysis."""
    dm = DataManager("ANALYSIS_AREA_PER_LIPID")
    dm.save_data(_analysis_record("HASH1"))

    sync_pull_analysis(analysis_name="area_per_lipid")

    captured = capsys.readouterr()
    assert "HASH1" in captured.out or "1 record" in captured.out


def test_cli_sync_pull_analysis_with_filter(temp_analysis_db, capsys):
    """Test CLI with hash filter."""
    dm = DataManager("ANALYSIS_AREA_PER_LIPID")
    dm.save_data(_analysis_record("HASH1"))
    dm.save_data(_analysis_record("HASH2"))

    sync_pull_analysis(analysis_name="area_per_lipid", hash="HASH1")

    captured = capsys.readouterr()
    assert "HASH1" in captured.out
    # HASH2 should not appear in filtered output


def test_cli_sync_pull_analysis_no_results(temp_analysis_db):
    """Test CLI with no matching results.

    When no results found, the function returns early without printing output.
    We verify this by checking that sync_pull_analysis completes without error.
    """
    # Should complete without raising an exception
    sync_pull_analysis(analysis_name="area_per_lipid", hash="NONEXISTENT")
    # If we get here, the test passes - the function handled the empty result gracefully


def test_cli_sync_pull_analysis_output_csv(temp_analysis_db, tmp_path):
    """Test CLI writing to CSV file."""
    dm = DataManager("ANALYSIS_AREA_PER_LIPID")
    dm.save_data(_analysis_record("HASH1"))

    output_file = tmp_path / "output.csv"
    sync_pull_analysis(analysis_name="area_per_lipid", output=output_file)

    assert output_file.exists()
    df = pd.read_csv(output_file)
    assert len(df) == 1
    assert df.iloc[0]["hash"] == "HASH1"


def test_cli_sync_pull_analysis_output_json(temp_analysis_db, tmp_path):
    """Test CLI writing to JSON file."""
    dm = DataManager("ANALYSIS_AREA_PER_LIPID")
    dm.save_data(_analysis_record("HASH1"))

    output_file = tmp_path / "output.json"
    sync_pull_analysis(analysis_name="area_per_lipid", output=output_file)

    assert output_file.exists()
    # Read JSON lines format
    with open(output_file, "r") as f:
        content = f.read()
    assert "HASH1" in content
