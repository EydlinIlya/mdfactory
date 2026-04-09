# ABOUTME: Helper functions for pulling artifact metadata from database
# ABOUTME: Queries artifact tables and formats results for display or export

import pandas as pd

from ..analysis.constants import ARTIFACT_DISPLAY_COLUMNS
from .pull_analysis import (
    pull_artifact,
    pull_overview,
)


def format_artifact_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Format artifact records for CLI display.

    Parameters
    ----------
    df : pd.DataFrame
        Artifact records

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with columns in preferred order
    """
    if df.empty:
        return df

    # Keep only columns that exist, then add any remaining
    display_columns = [c for c in ARTIFACT_DISPLAY_COLUMNS if c in df.columns]
    remaining = [c for c in df.columns if c not in display_columns]
    display_columns.extend(remaining)

    return df[display_columns]


def pull_artifact_overview(
    artifact_name: str | None = None,
    hash: str | None = None,
    simulation_type: str | None = None,
    status: str | None = None,
) -> pd.DataFrame:
    """Pull artifact overview data with optional filters.

    Parameters
    ----------
    artifact_name : str | None
        Filter by artifact name
    hash : str | None
        Filter by simulation hash
    simulation_type : str | None
        Filter by simulation type
    status : str | None
        Filter by status ("completed" or "not_yet_run")

    Returns
    -------
    pd.DataFrame
        Overview records for artifacts matching filters
    """
    df = pull_overview(
        hash=hash,
        simulation_type=simulation_type,
        item_type="artifact",
        item_name=artifact_name,
    )

    if df.empty:
        return df

    # Filter by status if specified
    if status is not None:
        df = df[df["status"] == status]

    return df


# Re-export for convenience
__all__ = [
    "format_artifact_summary",
    "pull_artifact",
    "pull_artifact_overview",
    "pull_overview",
]
