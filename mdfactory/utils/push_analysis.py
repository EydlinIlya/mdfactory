# ABOUTME: Helper functions for pushing analysis data to database
# ABOUTME: Discovers analysis results, serializes them, and uploads to analysis tables
"""Helper functions for pushing analysis data to database."""

import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ..analysis.constants import (
    ANALYSIS_COLUMNS,
    ARTIFACT_COLUMNS,
    OVERVIEW_COLUMNS,
)
from ..analysis.registry_queries import (
    get_all_analysis_names,
    get_all_artifact_names,
    get_analyses_for_simulation_type,
    get_analysis_table_name,
    get_artifact_table_name,
    get_artifacts_for_simulation_type,
)
from ..analysis.simulation import Simulation
from ..settings import Settings
from .data_manager import PLACEHOLDER_HASH, DataManager
from .db_operations import (
    init_csv_tables,
    init_foundry_tables,
    init_sqlite_tables,
    query_existing_hashes,
    upload_records,
)
from .push import discover_simulation_folders

__all__ = [
    "serialize_dataframe_to_csv",
    "deserialize_csv_to_dataframe",
    "prepare_analysis_record",
    "prepare_artifact_record",
    "prepare_overview_record",
    "query_existing_hashes",
    "upload_analysis_data",
    "update_overview_records",
    "discover_and_prepare_analysis_data",
    "push_analysis",
    "init_analysis_database",
    "init_artifact_database",
    "get_analysis_table_name",
    "get_artifact_table_name",
    "get_all_analysis_names",
    "get_all_artifact_names",
    "get_analyses_for_simulation_type",
    "get_artifacts_for_simulation_type",
]


def get_overview_placeholder() -> dict[str, Any]:
    """Create a placeholder record for ANALYSIS_OVERVIEW schema initialization."""
    return {
        "hash": PLACEHOLDER_HASH,
        "simulation_type": "bilayer",
        "directory": "/placeholder",
        "item_type": "analysis",
        "item_name": "placeholder",
        "status": "not_yet_run",
        "row_count": 0,
        "file_count": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def get_analysis_placeholder() -> dict[str, Any]:
    """Create a placeholder record for ANALYSIS_* table schema initialization.

    Note on data storage fields:
    - data_csv: Stores full serialized CSV data for database-only retrieval.
      Enables pulling complete analysis data without filesystem access.
    - data_path: Stores relative path to local parquet file (e.g., ".analysis/apl.parquet").
      Used for filesystem-based workflows when working with local files.

    Both fields serve complementary purposes: data_csv enables centralized database
    access while data_path references the canonical local storage location.
    """
    return {
        "hash": PLACEHOLDER_HASH,
        "directory": "/placeholder",
        "simulation_type": "bilayer",
        "row_count": 0,
        "columns": "[]",
        "data_csv": "",
        "data_path": ".analysis/placeholder.parquet",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def get_artifact_placeholder() -> dict[str, Any]:
    """Create a placeholder record for ARTIFACT_* table schema initialization."""
    return {
        "hash": PLACEHOLDER_HASH,
        "directory": "/placeholder",
        "simulation_type": "bilayer",
        "file_count": 0,
        "files": "[]",
        "checksums": "{}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def serialize_dataframe_to_csv(df: pd.DataFrame) -> str:
    """Serialize a DataFrame to CSV string.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to serialize

    Returns
    -------
    str
        CSV-formatted string

    """
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def deserialize_csv_to_dataframe(csv_string: str) -> pd.DataFrame:
    """Deserialize a CSV string to DataFrame.

    Parameters
    ----------
    csv_string : str
        CSV-formatted string

    Returns
    -------
    pd.DataFrame
        DataFrame

    """
    buffer = StringIO(csv_string)
    return pd.read_csv(buffer)


def prepare_analysis_record(
    sim: Simulation,
    analysis_name: str,
) -> dict[str, Any] | None:
    """Prepare a single analysis record for upload.

    Parameters
    ----------
    sim : Simulation
        Simulation instance
    analysis_name : str
        Analysis name

    Returns
    -------
    dict[str, Any] | None
        Record dict ready for database, or None if analysis not completed

    """
    completed_analyses = sim.list_analyses()
    if analysis_name not in completed_analyses:
        return None

    try:
        df = sim.load_analysis(analysis_name)
    except Exception as e:
        logger.warning(f"Failed to load analysis '{analysis_name}' for {sim.path.name}: {e}")
        return None

    # Serialize DataFrame to CSV for database storage
    # data_csv enables database-only retrieval without filesystem access
    data_csv = serialize_dataframe_to_csv(df)
    # data_path references the local parquet file location
    data_path = f".analysis/{analysis_name}.parquet"

    return {
        "hash": sim.build_input.hash,
        "directory": str(sim.path),
        "simulation_type": sim.build_input.simulation_type,
        "row_count": len(df),
        # columns is JSON-serialized to match ANALYSIS_COLUMNS schema
        "columns": json.dumps(df.columns.tolist()),
        "data_csv": data_csv,
        "data_path": data_path,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def prepare_artifact_record(
    sim: Simulation,
    artifact_name: str,
) -> dict[str, Any] | None:
    """Prepare a single artifact record for upload.

    Parameters
    ----------
    sim : Simulation
        Simulation instance
    artifact_name : str
        Artifact name

    Returns
    -------
    dict[str, Any] | None
        Record dict ready for database, or None if artifact not completed

    """
    completed_artifacts = sim.list_artifacts()
    if artifact_name not in completed_artifacts:
        return None

    try:
        entry = sim.registry.get_artifact_entry(artifact_name)
    except Exception as e:
        logger.warning(f"Failed to get artifact entry '{artifact_name}' for {sim.path.name}: {e}")
        return None

    files = entry.get("files", [])
    checksums = entry.get("checksums", {})

    return {
        "hash": sim.build_input.hash,
        "directory": str(sim.path),
        "simulation_type": sim.build_input.simulation_type,
        "file_count": len(files),
        # files and checksums are JSON-serialized to match ARTIFACT_COLUMNS schema
        "files": json.dumps(files),
        "checksums": json.dumps(checksums),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def prepare_overview_record(
    sim: Simulation,
    item_type: str,
    item_name: str,
    status: str,
    row_count: int = 0,
    file_count: int = 0,
) -> dict[str, Any]:
    """Prepare an overview record.

    Parameters
    ----------
    sim : Simulation
        Simulation instance
    item_type : str
        "analysis" or "artifact"
    item_name : str
        Name of the analysis or artifact
    status : str
        "completed" or "not_yet_run"
    row_count : int
        Row count for analyses (0 if not applicable)
    file_count : int
        File count for artifacts (0 if not applicable)

    Returns
    -------
    dict[str, Any]
        Record dict matching OVERVIEW_COLUMNS schema

    """
    return {
        "hash": sim.build_input.hash,
        "simulation_type": sim.build_input.simulation_type,
        "directory": str(sim.path),
        "item_type": item_type,
        "item_name": item_name,
        "status": status,
        "row_count": row_count,
        "file_count": file_count,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def upload_analysis_data(
    records: list[dict[str, Any]],
    table_name: str,
    force: bool = False,
    diff: bool = False,
) -> int:
    """Upload analysis records to database.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Records to upload
    table_name : str
        Table name
    force : bool
        Delete existing records before uploading
    diff : bool
        Skip records that already exist

    Returns
    -------
    int
        Number of records uploaded

    Raises
    ------
    ValueError
        If duplicates exist in default mode

    """
    return upload_records(records, table_name, ["hash"], force=force, diff=diff)


def update_overview_records(
    records: list[dict[str, Any]],
    force: bool = False,
    diff: bool = False,
) -> int:
    """Update overview table with records.

    Conflict behavior is controlled explicitly by the flags:
    - force=True: overwrite existing composite keys
    - diff=True: skip existing keys unless upgrading status (e.g. not_yet_run -> completed)
    - default: insert new keys and allow status upgrades, but raise on other duplicates

    Parameters
    ----------
    records : list[dict[str, Any]]
        Overview records
    force : bool
        Overwrite existing entries
    diff : bool
        Skip records that already exist, but allow status upgrades

    Returns
    -------
    int
        Number of records processed

    """
    if force and diff:
        raise ValueError("Cannot use both force and diff")
    if not records:
        return 0

    key_fields = ["hash", "item_type", "item_name"]

    if force:
        count = upload_records(records, "ANALYSIS_OVERVIEW", key_fields, force=force)
        logger.info(f"Updated {count} overview record(s)")
        return count

    # Default and diff modes both allow status upgrades (not_yet_run -> completed).
    # Only default mode raises on non-upgrade duplicates.
    dm = DataManager("ANALYSIS_OVERVIEW")
    existing_df = dm.load_data()
    existing_status: dict[tuple, str] = {}
    if not existing_df.empty and all(k in existing_df.columns for k in key_fields):
        for _, row in existing_df.iterrows():
            key = tuple(row[k] for k in key_fields)
            existing_status[key] = row.get("status", "")

    new_records = []
    upgrade_records = []
    skipped = 0
    duplicates = []
    for record in records:
        key = tuple(record.get(k) for k in key_fields)
        old_status = existing_status.get(key)
        if old_status is None:
            new_records.append(record)
        elif old_status != "completed" and record.get("status") == "completed":
            upgrade_records.append(record)
        elif diff:
            skipped += 1
        else:
            duplicates.append(key)

    if duplicates:
        logger.error(f"Found {len(duplicates)} duplicate overview key(s) in ANALYSIS_OVERVIEW")
        raise ValueError(
            f"{len(duplicates)} key(s) already exist in ANALYSIS_OVERVIEW. "
            f"Use --force to overwrite or --diff to skip."
        )

    if diff and skipped:
        logger.info(f"Skipped {skipped} overview record(s) (diff mode, no upgrade)")

    count = 0
    if new_records:
        count += upload_records(new_records, "ANALYSIS_OVERVIEW", key_fields, diff=True)
    if upgrade_records:
        count += upload_records(upgrade_records, "ANALYSIS_OVERVIEW", key_fields, force=True)

    logger.info(f"Updated {count} overview record(s)")
    return count


def discover_and_prepare_analysis_data(
    simulations: list[tuple[Path, Any]],
    analysis_name: str | None = None,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Discover analysis data from simulation folders.

    Parameters
    ----------
    simulations : list[tuple[Path, BuildInput]]
        List of (folder_path, build_input) tuples
    analysis_name : str | None
        Specific analysis to discover, or None for all

    Returns
    -------
    tuple[dict[str, list[dict]], list[dict]]
        (analysis_records_by_table, overview_records)
        analysis_records_by_table maps table_name to list of records

    """
    analysis_records: dict[str, list[dict]] = {}
    overview_records: list[dict] = []

    for folder, build_input in simulations:
        try:
            sim = Simulation(folder, build_input=build_input, trajectory_file=None)
        except Exception as e:
            logger.warning(f"Failed to create Simulation for {folder}: {e}")
            continue

        sim_type = sim.build_input.simulation_type

        # Get analyses to check
        if analysis_name:
            analyses_to_check = [analysis_name]
        else:
            analyses_to_check = get_analyses_for_simulation_type(sim_type)

        # Process analyses
        completed_analyses = set(sim.list_analyses())
        for name in analyses_to_check:
            if name in completed_analyses:
                # Completed analysis - prepare full record
                record = prepare_analysis_record(sim, name)
                if record:
                    table_name = get_analysis_table_name(name)
                    if table_name not in analysis_records:
                        analysis_records[table_name] = []
                    analysis_records[table_name].append(record)

                    # Add completed overview entry
                    overview_records.append(
                        prepare_overview_record(
                            sim, "analysis", name, "completed", row_count=record["row_count"]
                        )
                    )
            else:
                # Not completed - add not_yet_run overview entry
                overview_records.append(
                    prepare_overview_record(sim, "analysis", name, "not_yet_run")
                )

    return analysis_records, overview_records


def push_analysis(
    source: Path | None = None,
    csv: Path | None = None,
    csv_root: Path | None = None,
    analysis_name: str | None = None,
    force: bool = False,
    diff: bool = False,
) -> dict[str, int]:
    """Push analysis data to database.

    Parameters
    ----------
    source : Path | None
        Directory path, glob pattern, or summary YAML file (auto-detected)
    csv : Path | None
        CSV file with build specifications
    csv_root : Path | None
        Root directory for CSV hash search
    analysis_name : str | None
        Specific analysis to push, or None for all
    force : bool
        Delete existing records before uploading
    diff : bool
        Skip records that already exist

    Returns
    -------
    dict[str, int]
        {table_name: count_uploaded}

    """
    # Discover simulations
    simulations = discover_simulation_folders(source=source, csv=csv, csv_root=csv_root)
    logger.info(f"Discovered {len(simulations)} simulation(s)")

    if not simulations:
        logger.warning("No simulations found")
        return {}

    # Prepare records
    data_records, overview_records = discover_and_prepare_analysis_data(
        simulations,
        analysis_name=analysis_name,
    )

    results = {}

    # Upload analysis/artifact data
    for table_name, records in data_records.items():
        count = upload_analysis_data(records, table_name, force=force, diff=diff)
        results[table_name] = count

    # Update overview table
    if overview_records:
        count = update_overview_records(overview_records, force=force, diff=diff)
        results["ANALYSIS_OVERVIEW"] = count

    return results


def _build_analysis_table_list() -> list[tuple[str, dict, list[str]]]:
    """Build table list for analysis database initialization.

    Excludes ANALYSIS_OVERVIEW because both ``init_analysis_database`` and
    ``init_artifact_database`` handle it separately via ``_init_overview_table``.

    Returns
    -------
    list[tuple[str, dict, list[str]]]
        List of (table_name, placeholder_record, columns) tuples

    """
    tables: list[tuple[str, dict, list[str]]] = []
    for name in get_all_analysis_names():
        table_name = get_analysis_table_name(name)
        tables.append((table_name, get_analysis_placeholder(), ANALYSIS_COLUMNS))
    return tables


def _build_artifact_table_list() -> list[tuple[str, dict, list[str]]]:
    """Build table list for artifact database initialization.

    Intentionally excludes ANALYSIS_OVERVIEW so artifact-only reset cannot
    remove analysis overview state.

    Returns
    -------
    list[tuple[str, dict, list[str]]]
        List of (table_name, placeholder_record, columns) tuples

    """
    tables: list[tuple[str, dict, list[str]]] = []
    for name in get_all_artifact_names():
        table_name = get_artifact_table_name(name)
        tables.append((table_name, get_artifact_placeholder(), ARTIFACT_COLUMNS))
    return tables


def _init_overview_table(database_type: str) -> dict[str, bool]:
    """Ensure ANALYSIS_OVERVIEW exists for the active backend.

    Parameters
    ----------
    database_type : str
        Backend type ("sqlite", "csv", or "foundry")

    Returns
    -------
    dict[str, bool]
        {table_name: was_created}

    """
    overview_table = [("ANALYSIS_OVERVIEW", get_overview_placeholder(), OVERVIEW_COLUMNS)]
    if database_type == "sqlite":
        return init_sqlite_tables(overview_table, reset=False)
    elif database_type == "csv":
        return init_csv_tables(overview_table, reset=False)
    elif database_type == "foundry":
        return init_foundry_tables(overview_table, reset=False)
    raise ValueError(f"Unsupported database type: {database_type}")


def _clear_overview_item_type(item_type: str) -> None:
    """Delete rows from ANALYSIS_OVERVIEW for a specific item type.

    Parameters
    ----------
    item_type : str
        Item type to clear ("analysis" or "artifact")

    """
    dm = DataManager("ANALYSIS_OVERVIEW")
    dm.delete_data({"item_type": item_type})


def init_analysis_database(
    reset: bool = False,
) -> dict[str, bool]:
    """Initialize analysis database tables.

    Creates tables (SQLite) or datasets (Foundry) for all registered
    analysis types and the overview table. On reset, only overview rows
    with ``item_type='analysis'`` are cleared.

    Parameters
    ----------
    reset : bool
        Recreate tables even if they exist

    Returns
    -------
    dict[str, bool]
        {table_name: was_created}

    """
    config = Settings()
    database_type = config.config["database"]["TYPE"]
    tables = _build_analysis_table_list()

    if database_type == "sqlite":
        results = init_sqlite_tables(tables, reset=reset)
    elif database_type == "csv":
        results = init_csv_tables(tables, reset=reset)
    elif database_type == "foundry":
        results = init_foundry_tables(tables, reset=reset)
    else:
        raise ValueError(f"Unsupported database type: {database_type}")

    # Always ensure overview exists and only clear analysis rows on reset.
    overview_results = _init_overview_table(database_type)
    results.update(overview_results)
    if reset:
        _clear_overview_item_type("analysis")

    return results


def init_artifact_database(
    reset: bool = False,
) -> dict[str, bool]:
    """Initialize artifact database tables.

    Creates tables (SQLite) or datasets (Foundry) for all registered
    artifact types. On reset, only overview rows with
    ``item_type='artifact'`` are cleared.

    Parameters
    ----------
    reset : bool
        Recreate tables even if they exist

    Returns
    -------
    dict[str, bool]
        {table_name: was_created}

    """
    config = Settings()
    database_type = config.config["database"]["TYPE"]
    tables = _build_artifact_table_list()

    if database_type == "sqlite":
        results = init_sqlite_tables(tables, reset=reset)
    elif database_type == "csv":
        results = init_csv_tables(tables, reset=reset)
    elif database_type == "foundry":
        results = init_foundry_tables(tables, reset=reset)
    else:
        raise ValueError(f"Unsupported database type: {database_type}")

    # Keep overview table synchronized while preserving analysis rows.
    _init_overview_table(database_type)
    if reset:
        _clear_overview_item_type("artifact")

    return results
