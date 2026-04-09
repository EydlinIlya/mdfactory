# ABOUTME: Tests for the AnalysisRegistry class, including registration, discovery,
# ABOUTME: and execution of analysis plugins.
"""Tests for AnalysisRegistry class."""

from pathlib import Path

import pandas as pd
import pytest

from mdfactory.analysis.registry import AnalysisRegistry


@pytest.fixture
def temp_analysis_dir(tmp_path):
    """Create temporary .analysis directory."""
    analysis_dir = tmp_path / ".analysis"
    analysis_dir.mkdir()
    return analysis_dir


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for analysis data."""
    return pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "category": ["A", "B", "A", "B", "A"],
        }
    )


@pytest.fixture
def registry(temp_analysis_dir):
    """Create AnalysisRegistry instance."""
    return AnalysisRegistry(temp_analysis_dir)


def test_init(temp_analysis_dir):
    """Test AnalysisRegistry initialization."""
    registry = AnalysisRegistry(temp_analysis_dir)
    assert registry.analysis_dir == temp_analysis_dir
    assert registry._registry is None


def test_registry_path(temp_analysis_dir):
    """Test registry_path property."""
    registry = AnalysisRegistry(temp_analysis_dir)
    expected = temp_analysis_dir / "metadata.json"
    assert registry.registry_path == expected


def test_load_nonexistent_returns_default(registry):
    """Test loading when registry file doesn't exist returns default."""
    result = registry.load()
    assert result["schema_version"] == "1.0"
    assert result["analyses"] == {}
    assert result["artifacts"] == {}
    assert registry._registry == result


def test_load_creates_default_registry(registry):
    """Test that load() creates default registry structure."""
    registry.load()
    assert registry._registry is not None
    assert "schema_version" in registry._registry
    assert "analyses" in registry._registry
    assert "artifacts" in registry._registry


def test_save_creates_directory(tmp_path):
    """Test that save() creates .analysis directory if needed."""
    analysis_dir = tmp_path / "new_dir" / ".analysis"
    registry = AnalysisRegistry(analysis_dir)
    registry.save()

    assert analysis_dir.exists()
    assert (analysis_dir / "metadata.json").exists()


def test_load_save_roundtrip(registry, temp_analysis_dir):
    """Test loading and saving registry preserves data."""
    registry.load()
    registry._registry["analyses"]["test_analysis"] = {
        "filename": "test.parquet",
        "row_count": 100,
        "columns": ["a", "b"],
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "extras": {},
    }
    registry.save()

    # Load with new instance
    registry2 = AnalysisRegistry(temp_analysis_dir)
    registry2.load()

    assert registry2._registry is not None
    assert "test_analysis" in registry2._registry["analyses"]
    assert registry2._registry["analyses"]["test_analysis"]["row_count"] == 100


def test_load_corrupted_registry_returns_default(registry, temp_analysis_dir):
    """Test loading corrupted JSON returns default with warning."""
    # Create corrupted JSON file
    (temp_analysis_dir / "metadata.json").write_text("{ invalid json")

    result = registry.load()

    assert result["schema_version"] == "1.0"
    assert result["analyses"] == {}
    assert result["artifacts"] == {}


def test_add_entry(registry, sample_dataframe):
    """Test adding new analysis entry."""
    registry.add_entry("test_analysis", sample_dataframe, custom_field="value")
    registry.load()

    entry = registry._registry["analyses"]["test_analysis"]
    assert entry["filename"] == "test_analysis.parquet"
    assert entry["row_count"] == 5
    assert entry["columns"] == ["time", "value", "category"]
    assert "created_at" in entry
    assert "updated_at" in entry
    assert entry["extras"]["custom_field"] == "value"


def test_add_entry_duplicate_raises(registry, sample_dataframe):
    """Test adding duplicate analysis raises ValueError."""
    registry.add_entry("test_analysis", sample_dataframe)

    with pytest.raises(ValueError, match="already exists"):
        registry.add_entry("test_analysis", sample_dataframe)


def test_update_entry_creates_if_missing(registry, sample_dataframe):
    """Test update_entry creates entry if it doesn't exist."""
    registry.update_entry("new_analysis", sample_dataframe)

    assert "new_analysis" in registry._registry["analyses"]
    assert registry._registry["analyses"]["new_analysis"]["row_count"] == 5


def test_update_entry_preserves_created_at(registry, sample_dataframe):
    """Test update_entry preserves created_at timestamp."""
    # Add entry
    registry.add_entry("test_analysis", sample_dataframe)
    original_created_at = registry._registry["analyses"]["test_analysis"]["created_at"]

    # Update entry with different data
    new_df = pd.DataFrame({"a": [1, 2, 3]})
    registry.update_entry("test_analysis", new_df)

    entry = registry._registry["analyses"]["test_analysis"]
    assert entry["created_at"] == original_created_at
    assert entry["row_count"] == 3  # Updated
    assert entry["columns"] == ["a"]  # Updated


def test_update_entry_updates_timestamp(registry, sample_dataframe):
    """Test update_entry updates the updated_at timestamp."""
    registry.add_entry("test_analysis", sample_dataframe)
    original_updated_at = registry._registry["analyses"]["test_analysis"]["updated_at"]

    # Small delay to ensure different timestamp (not strictly necessary with UTC)
    new_df = pd.DataFrame({"a": [1, 2, 3]})
    registry.update_entry("test_analysis", new_df)

    # updated_at should be same or later (may be same if very fast)
    new_updated_at = registry._registry["analyses"]["test_analysis"]["updated_at"]
    assert new_updated_at >= original_updated_at


def test_get_entry(registry, sample_dataframe):
    """Test retrieving analysis entry."""
    registry.add_entry("test_analysis", sample_dataframe)

    entry = registry.get_entry("test_analysis")

    assert entry["filename"] == "test_analysis.parquet"
    assert entry["row_count"] == 5


def test_get_entry_missing_raises(registry):
    """Test get_entry with missing analysis raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        registry.get_entry("nonexistent")


def test_list_analyses_empty(registry):
    """Test list_analyses returns empty list for new registry."""
    assert registry.list_analyses() == []


def test_list_analyses(registry, sample_dataframe):
    """Test list_analyses returns sorted analysis names."""
    registry.add_entry("zebra", sample_dataframe)
    registry.add_entry("alpha", sample_dataframe)
    registry.add_entry("beta", sample_dataframe)

    analyses = registry.list_analyses()

    assert analyses == ["alpha", "beta", "zebra"]


def test_artifact_entry_roundtrip(registry):
    """Test adding and listing artifact entries."""
    files = ["artifacts/last_frame_pdb/last_frame.pdb"]
    checksums = {files[0]: "deadbeef"}

    registry.add_artifact_entry("last_frame_pdb", files, checksums, purpose="test")

    assert registry.list_artifacts() == ["last_frame_pdb"]
    entry = registry.get_artifact_entry("last_frame_pdb")
    assert entry["files"] == files
    assert entry["checksums"] == checksums
    assert entry["extras"]["purpose"] == "test"


def test_check_integrity_valid(registry, temp_analysis_dir, sample_dataframe):
    """Test check_integrity returns valid for matching registry and files."""
    registry.add_entry("test_analysis", sample_dataframe)
    registry.save()

    # Create matching parquet file
    sample_dataframe.to_parquet(temp_analysis_dir / "test_analysis.parquet")

    result = registry.check_integrity()

    assert result["valid"] is True
    assert result["missing_files"] == []
    assert result["extra_files"] == []
    assert result["row_count_mismatches"] == []
    assert result["artifact_missing_files"] == []
    assert result["artifact_checksum_mismatches"] == []


def test_check_integrity_missing_files(registry, sample_dataframe):
    """Test check_integrity detects missing parquet files."""
    registry.add_entry("missing_analysis", sample_dataframe)

    # Don't create the parquet file

    result = registry.check_integrity()

    assert result["valid"] is False
    assert "missing_analysis" in result["missing_files"]


def test_check_integrity_extra_files(registry, temp_analysis_dir, sample_dataframe):
    """Test check_integrity detects orphaned parquet files."""
    # Create parquet file not in registry
    sample_dataframe.to_parquet(temp_analysis_dir / "orphan.parquet")

    result = registry.check_integrity()

    assert result["valid"] is False
    assert "orphan.parquet" in result["extra_files"]


def test_check_integrity_row_count_mismatch(registry, temp_analysis_dir, sample_dataframe):
    """Test check_integrity detects row count mismatches."""
    registry.add_entry("test_analysis", sample_dataframe)
    registry.save()

    # Create parquet with different row count
    different_df = pd.DataFrame({"time": [0, 1]})  # Only 2 rows instead of 5
    different_df.to_parquet(temp_analysis_dir / "test_analysis.parquet")

    result = registry.check_integrity()

    assert result["valid"] is False
    assert len(result["row_count_mismatches"]) == 1
    mismatch = result["row_count_mismatches"][0]
    assert mismatch["name"] == "test_analysis"
    assert mismatch["expected"] == 5
    assert mismatch["actual"] == 2


def test_check_integrity_multiple_issues(registry, temp_analysis_dir, sample_dataframe):
    """Test check_integrity detects multiple issues simultaneously."""
    registry.add_entry("analysis1", sample_dataframe)
    registry.add_entry("analysis2", sample_dataframe)

    # Create only analysis1 with wrong row count
    wrong_df = pd.DataFrame({"x": [1]})
    wrong_df.to_parquet(temp_analysis_dir / "analysis1.parquet")

    # Create orphaned file
    sample_dataframe.to_parquet(temp_analysis_dir / "orphan.parquet")

    result = registry.check_integrity()

    assert result["valid"] is False
    assert "analysis2" in result["missing_files"]  # Missing file
    assert "orphan.parquet" in result["extra_files"]  # Extra file
    assert len(result["row_count_mismatches"]) == 1  # Row count mismatch


def test_create_default_registry(registry):
    """Test _create_default_registry creates correct structure."""
    default = registry._create_default_registry()

    assert default["schema_version"] == "1.0"
    assert default["analyses"] == {}
    assert default["artifacts"] == {}


def test_extract_metadata(registry, sample_dataframe):
    """Test _extract_metadata extracts correct info."""
    metadata = registry._extract_metadata(sample_dataframe)

    assert metadata["row_count"] == 5
    assert metadata["columns"] == ["time", "value", "category"]


def test_get_timestamp_format(registry):
    """Test _get_timestamp returns ISO 8601 format."""
    timestamp = registry._get_timestamp()

    # Check it's a valid ISO 8601 string with timezone
    assert "T" in timestamp
    assert "+" in timestamp or "Z" in timestamp or timestamp.endswith("+00:00")


def test_registry_with_string_path(tmp_path):
    """Test AnalysisRegistry accepts string path."""
    path_str = str(tmp_path / ".analysis")
    registry = AnalysisRegistry(path_str)

    assert isinstance(registry.analysis_dir, Path)
