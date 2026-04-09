# ABOUTME: Helper functions for pulling analysis data from database
# ABOUTME: Queries analysis tables and formats results for display or export
"""Helper functions for pulling analysis data from database."""

from typing import Any, TypedDict

import pandas as pd
from loguru import logger

from ..analysis.constants import OVERVIEW_COLUMNS
from ..analysis.registry_queries import (
    get_all_analysis_names,
    get_all_artifact_names,
    get_analysis_table_name,
    get_artifact_table_name,
)
from .data_manager import DataManager
from .db_operations import drop_placeholder
from .push_analysis import deserialize_csv_to_dataframe


class AvailableTables(TypedDict):
    """Available analysis/artifact table names and overview-table presence."""

    analyses: list[str]
    artifacts: list[str]
    overview: bool


def list_available_tables() -> AvailableTables:
    """List available analysis and artifact tables that exist in database.

    Returns
    -------
    AvailableTables
        Mapping with analysis names, artifact names, and overview-table presence.

    """
    available = {"analyses": [], "artifacts": [], "overview": False}

    # Check analysis tables
    for name in get_all_analysis_names():
        table_name = get_analysis_table_name(name)
        try:
            dm = DataManager(table_name)
            if dm.data_source.table_exists:
                available["analyses"].append(name)
        except FileNotFoundError:
            pass

    # Check artifact tables
    for name in get_all_artifact_names():
        table_name = get_artifact_table_name(name)
        try:
            dm = DataManager(table_name)
            if dm.data_source.table_exists:
                available["artifacts"].append(name)
        except FileNotFoundError:
            pass

    # Check overview table
    try:
        dm = DataManager("ANALYSIS_OVERVIEW")
        if dm.data_source.table_exists:
            available["overview"] = True
    except FileNotFoundError:
        pass

    return available


def _query_table(table_name: str, **filters: Any) -> pd.DataFrame:
    """Query a table with optional filters, removing placeholders.

    Parameters
    ----------
    table_name : str
        Table/database name to query
    **filters
        Field-value pairs to filter by (None values are ignored)

    Returns
    -------
    pd.DataFrame
        Records matching filters with placeholders removed

    """
    dm = DataManager(table_name)
    conditions = {k: v for k, v in filters.items() if v is not None}

    if conditions:
        df = dm.query_data(conditions)
    else:
        df = dm.load_data()

    return drop_placeholder(df)


def pull_overview(
    hash: str | None = None,
    simulation_type: str | None = None,
    item_type: str | None = None,
    item_name: str | None = None,
) -> pd.DataFrame:
    """Pull data from overview table with optional filters.

    Parameters
    ----------
    hash : str | None
        Filter by simulation hash
    simulation_type : str | None
        Filter by simulation type
    item_type : str | None
        Filter by "analysis" or "artifact"
    item_name : str | None
        Filter by item name

    Returns
    -------
    pd.DataFrame
        Overview records matching filters

    """
    return _query_table(
        "ANALYSIS_OVERVIEW",
        hash=hash,
        simulation_type=simulation_type,
        item_type=item_type,
        item_name=item_name,
    )


def pull_analysis(
    analysis_name: str,
    hash: str | None = None,
    simulation_type: str | None = None,
    decode_data: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Pull analysis data from a specific analysis table.

    Parameters
    ----------
    analysis_name : str
        Analysis name (e.g., 'area_per_lipid')
    hash : str | None
        Filter by simulation hash
    simulation_type : str | None
        Filter by simulation type
    decode_data : bool
        If True, decode data_csv column to actual DataFrames.
        Returns a dict with decoded data instead of DataFrame.

    Returns
    -------
    pd.DataFrame | dict[str, pd.DataFrame]
        Analysis records matching filters, or decoded data if decode_data=True

    """
    table_name = get_analysis_table_name(analysis_name)
    df = _query_table(table_name, hash=hash, simulation_type=simulation_type)

    if decode_data:
        return decode_analysis_data(df)

    return df


def pull_artifact(
    artifact_name: str,
    hash: str | None = None,
    simulation_type: str | None = None,
) -> pd.DataFrame:
    """Pull artifact metadata from a specific artifact table.

    Parameters
    ----------
    artifact_name : str
        Artifact name (e.g., 'bilayer_snapshot')
    hash : str | None
        Filter by simulation hash
    simulation_type : str | None
        Filter by simulation type

    Returns
    -------
    pd.DataFrame
        Artifact records matching filters

    """
    table_name = get_artifact_table_name(artifact_name)
    return _query_table(table_name, hash=hash, simulation_type=simulation_type)


def decode_analysis_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Decode data_csv column from analysis records.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with data_csv column

    Returns
    -------
    dict[str, pd.DataFrame]
        {hash: decoded_dataframe}

    """
    if df.empty or "data_csv" not in df.columns:
        return {}

    result = {}
    for _, row in df.iterrows():
        try:
            decoded = deserialize_csv_to_dataframe(row["data_csv"])
            result[row["hash"]] = decoded
        except Exception as e:
            logger.warning(f"Failed to decode data for hash {row['hash']}: {e}")

    return result


def format_analysis_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Format analysis records for CLI display.

    Excludes large data columns (data_csv) for readable output.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis records

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame without large columns

    """
    if df.empty:
        return df

    # Columns to exclude from summary display
    exclude_columns = ["data_csv"]

    # Filter columns
    display_columns = [c for c in df.columns if c not in exclude_columns]
    return df[display_columns]


def format_overview_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Format overview records for CLI display.

    Parameters
    ----------
    df : pd.DataFrame
        Overview records

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame

    """
    if df.empty:
        return df

    # Use OVERVIEW_COLUMNS from constants for consistent column ordering.
    # The column order matches the schema definition in analysis.constants.
    preferred_order = OVERVIEW_COLUMNS

    # Keep only columns that exist
    display_columns = [c for c in preferred_order if c in df.columns]
    return df[display_columns]


def pull_systems_with_analyses(
    analysis_name: str | None = None,
    status: str | None = None,
    simulation_type: str | None = None,
) -> pd.DataFrame:
    """Pull summary of which systems have which analyses completed.

    Joins overview data to provide a pivot-like view.

    Parameters
    ----------
    analysis_name : str | None
        Filter to specific analysis
    status : str | None
        Filter by status ("completed" or "not_yet_run")
    simulation_type : str | None
        Filter by simulation type

    Returns
    -------
    pd.DataFrame
        Summary with hash, simulation_type, and analysis completion status

    """
    # Pull overview data filtered to analyses only
    df = pull_overview(
        simulation_type=simulation_type,
        item_type="analysis",
        item_name=analysis_name,
    )

    if df.empty:
        return df

    # Filter by status if specified
    if status is not None:
        df = df[df["status"] == status]

    return format_overview_summary(df)
