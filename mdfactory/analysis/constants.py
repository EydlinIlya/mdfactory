# ABOUTME: Centralized constants for analysis schemas and simulation status
# ABOUTME: Contains SimulationStatus enum and schema column definitions

from enum import Enum


class SimulationStatus(str, Enum):
    """Simulation status hierarchy enum.

    Status progresses through these stages in order:
    BUILD -> EQUILIBRATED -> PRODUCTION -> COMPLETED

    Each status indicates what output files exist:
    - BUILD: Valid build folder with YAML, but no simulation outputs
    - EQUILIBRATED: min.gro, nvt.gro, npt.gro all exist
    - PRODUCTION: prod.xtc exists (but no prod.gro)
    - COMPLETED: prod.gro exists
    """

    BUILD = "build"
    EQUILIBRATED = "equilibrated"
    PRODUCTION = "production"
    COMPLETED = "completed"

    @classmethod
    def get_order(cls) -> list[str]:
        """Get status values in hierarchy order (lowest to highest)."""
        return [s.value for s in cls]

    @classmethod
    def get_index(cls, status: str) -> int:
        """Get the index of a status in the hierarchy.

        Parameters
        ----------
        status : str
            Status string value

        Returns
        -------
        int
            Index in hierarchy (0 = BUILD, 3 = COMPLETED)

        Raises
        ------
        ValueError
            If status is not a valid status value
        """
        order = cls.get_order()
        if status not in order:
            raise ValueError(f"Invalid status '{status}'. Must be one of: {order}")
        return order.index(status)


# Status order list for backwards compatibility
# Prefer using SimulationStatus enum in new code
STATUS_ORDER = SimulationStatus.get_order()


# Equilibration output files (Gromacs engine)
# NOTE: completed_file and trajectory_file are user-configurable via Simulation/SimulationStore
# constructor parameters. Only equilibration_files is centralized here as a default.
EQUILIBRATION_FILES = ["min.gro", "nvt.gro", "npt.gro"]

# Summary display columns for CLI (subset of RUN_DATABASE_COLUMNS)
SUMMARY_COLUMNS = ["hash", "simulation_type", "parametrization", "status", "directory"]

# Artifact display column order
ARTIFACT_DISPLAY_COLUMNS = [
    "hash",
    "simulation_type",
    "file_count",
    "files",
    "checksums",
    "directory",
    "timestamp_utc",
]


# Schema definitions for database tables

OVERVIEW_COLUMNS = [
    "hash",
    "simulation_type",
    "directory",
    "item_type",
    "item_name",
    "status",
    "row_count",
    "file_count",
    "updated_at",
]

ANALYSIS_COLUMNS = [
    "hash",
    "directory",
    "simulation_type",
    "row_count",
    "columns",
    "data_csv",  # Full serialized data for database-only retrieval
    "data_path",  # Relative path to local parquet file
    "timestamp_utc",
]

ARTIFACT_COLUMNS = [
    "hash",
    "directory",
    "simulation_type",
    "file_count",
    "files",
    "checksums",
    "timestamp_utc",
]

# Schema definition for RUN_DATABASE
RUN_DATABASE_COLUMNS = [
    "hash",
    "engine",
    "parametrization",
    "simulation_type",
    "input_data",
    "input_data_type",
    "directory",
    "status",
    "timestamp_utc",
]


# Foundry schema polling configuration
SCHEMA_POLL_TIMEOUT_SECONDS = 90
SCHEMA_POLL_INTERVAL_SECONDS = 2
