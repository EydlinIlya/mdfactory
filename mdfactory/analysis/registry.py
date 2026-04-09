# ABOUTME: Per-simulation analysis registry stored as JSON
# ABOUTME: Tracks which analyses have been run and stores their results
"""Analysis registry management for simulation analysis storage."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def save_after(method):
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self.save()
        return result

    return wrapper


def ensure_load_before(method):
    def wrapper(self, *args, **kwargs):
        if self._registry is None:
            self.load()
        return method(self, *args, **kwargs)

    return wrapper


class AnalysisRegistry:
    """Manages the analysis registry (.analysis/metadata.json) for a simulation.

    The registry tracks analysis results stored as Parquet files, maintaining
    metadata about each analysis including row counts, columns, and timestamps.

    Attributes
    ----------
    analysis_dir : Path
        Path to .analysis directory
    _registry : dict[str, Any] | None
        Dict holding the in-memory registry state

    """

    SCHEMA_VERSION = "1.0"
    REGISTRY_FILENAME = "metadata.json"

    def __init__(self, analysis_dir: Path | str):
        """Initialize registry for given .analysis directory.

        Does not auto-load; call load() explicitly to read from disk.

        Parameters
        ----------
        analysis_dir : Path | str
            Path to .analysis directory

        """
        self.analysis_dir = Path(analysis_dir)
        self._registry: dict[str, Any] | None = None

    @property
    def registry_path(self) -> Path:
        """Path to metadata.json file."""
        return self.analysis_dir / self.REGISTRY_FILENAME

    def load(self) -> dict[str, Any]:
        """Load registry from disk.

        If file doesn't exist or is corrupted, returns default empty registry
        with a warning.

        Returns
        -------
        dict[str, Any]
            The registry dict with 'schema_version' and 'analyses' keys

        """
        from loguru import logger

        if not self.registry_path.exists():
            logger.debug(f"Registry file not found at {self.registry_path}, creating default")
            self._registry = self._create_default_registry()
            self._ensure_registry_keys()
            assert self._registry is not None
            return self._registry

        try:
            with open(self.registry_path, "r") as f:
                self._registry = json.load(f)
            self._ensure_registry_keys()
            assert self._registry is not None
            logger.debug(f"Loaded registry from {self.registry_path}")
            return self._registry
        except json.JSONDecodeError as e:
            logger.warning(
                f"Registry file at {self.registry_path} is corrupted: {e}. "
                f"Returning default empty registry."
            )
            self._registry = self._create_default_registry()
            self._ensure_registry_keys()
            return self._registry

    @ensure_load_before
    def save(self) -> None:
        """Save current registry state to disk.

        Creates .analysis directory if it doesn't exist.
        """
        from loguru import logger

        # Create directory if needed
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_registry_keys()

        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)

        logger.debug(f"Saved registry to {self.registry_path}")

    @ensure_load_before
    @save_after
    def add_entry(self, name: str, df: pd.DataFrame, **extras) -> None:
        """Add new analysis entry to registry.

        Parameters
        ----------
        name : str
            Analysis name (without .parquet extension)
        df : pd.DataFrame
            DataFrame to extract metadata from
        **extras
            Additional metadata to store in 'extras' field

        Raises
        ------
        ValueError
            If analysis with this name already exists

        """
        from loguru import logger

        assert self._registry is not None
        if name in self._registry["analyses"]:
            raise ValueError(f"Analysis '{name}' already exists in registry")

        timestamp = self._get_timestamp()
        metadata = self._extract_metadata(df)

        self._registry["analyses"][name] = {
            "filename": f"{name}.parquet",
            "row_count": metadata["row_count"],
            "columns": metadata["columns"],
            "created_at": timestamp,
            "updated_at": timestamp,
            "extras": extras,
        }

        logger.debug(f"Added analysis '{name}' to registry")

    @ensure_load_before
    @save_after
    def update_entry(self, name: str, df: pd.DataFrame, **extras) -> None:
        """Update existing analysis entry or create if doesn't exist.

        Preserves 'created_at' timestamp if entry exists, updates 'updated_at'.

        Parameters
        ----------
        name : str
            Analysis name (without .parquet extension)
        df : pd.DataFrame
            DataFrame to extract metadata from
        **extras
            Additional metadata to store in 'extras' field

        """
        from loguru import logger

        assert self._registry is not None
        timestamp = self._get_timestamp()
        metadata = self._extract_metadata(df)

        # Preserve created_at if entry exists
        created_at = timestamp
        if name in self._registry["analyses"]:
            created_at = self._registry["analyses"][name].get("created_at", timestamp)

        self._registry["analyses"][name] = {
            "filename": f"{name}.parquet",
            "row_count": metadata["row_count"],
            "columns": metadata["columns"],
            "created_at": created_at,
            "updated_at": timestamp,
            "extras": extras,
        }

        logger.debug(f"Updated analysis '{name}' in registry")

    @ensure_load_before
    def get_entry(self, name: str) -> dict[str, Any]:
        """Retrieve analysis entry by name.

        Parameters
        ----------
        name : str
            Analysis name

        Returns
        -------
        dict[str, Any]
            Dict with analysis metadata

        Raises
        ------
        KeyError
            If analysis not found in registry

        """
        assert self._registry is not None
        if name not in self._registry["analyses"]:
            raise KeyError(f"Analysis '{name}' not found in registry")

        return self._registry["analyses"][name]

    @ensure_load_before
    def list_analyses(self) -> list[str]:
        """Return sorted list of analysis names.

        Returns
        -------
        list[str]
            Sorted list of analysis names

        """
        assert self._registry is not None
        return sorted(self._registry["analyses"].keys())

    @ensure_load_before
    def list_artifacts(self) -> list[str]:
        """Return sorted list of artifact names.

        Returns
        -------
        list[str]
            Sorted list of artifact names

        """
        assert self._registry is not None
        return sorted(self._registry["artifacts"].keys())

    @ensure_load_before
    @save_after
    def add_artifact_entry(
        self,
        name: str,
        files: list[str],
        checksums: dict[str, str],
        **extras,
    ) -> None:
        """Add a new artifact entry to the registry.

        Parameters
        ----------
        name : str
            Artifact name
        files : list[str]
            Relative file paths under .analysis
        checksums : dict[str, str]
            Mapping of relative paths to sha256 checksums
        **extras
            Additional metadata to store

        """
        assert self._registry is not None
        if name in self._registry["artifacts"]:
            raise ValueError(f"Artifact '{name}' already exists in registry")

        timestamp = self._get_timestamp()
        self._registry["artifacts"][name] = {
            "files": files,
            "checksums": checksums,
            "created_at": timestamp,
            "updated_at": timestamp,
            "extras": extras,
        }

    @ensure_load_before
    @save_after
    def update_artifact_entry(
        self,
        name: str,
        files: list[str],
        checksums: dict[str, str],
        **extras,
    ) -> None:
        """Update or create an artifact entry in the registry.

        Parameters
        ----------
        name : str
            Artifact name
        files : list[str]
            Relative file paths under .analysis
        checksums : dict[str, str]
            Mapping of relative paths to sha256 checksums
        **extras
            Additional metadata to store

        """
        assert self._registry is not None
        timestamp = self._get_timestamp()
        created_at = timestamp
        if name in self._registry["artifacts"]:
            created_at = self._registry["artifacts"][name].get("created_at", timestamp)

        self._registry["artifacts"][name] = {
            "files": files,
            "checksums": checksums,
            "created_at": created_at,
            "updated_at": timestamp,
            "extras": extras,
        }

    @ensure_load_before
    def get_artifact_entry(self, name: str) -> dict[str, Any]:
        """Retrieve artifact entry by name.

        Parameters
        ----------
        name : str
            Artifact name

        Returns
        -------
        dict[str, Any]
            Dict with artifact metadata

        """
        assert self._registry is not None
        if name not in self._registry["artifacts"]:
            raise KeyError(f"Artifact '{name}' not found in registry")

        return self._registry["artifacts"][name]

    @ensure_load_before
    @save_after
    def remove_artifact_entry(self, name: str) -> None:
        """Remove artifact entry from registry.

        Parameters
        ----------
        name : str
            Artifact name to remove

        """
        assert self._registry is not None
        if name not in self._registry["artifacts"]:
            raise KeyError(f"Artifact '{name}' not found in registry")

        del self._registry["artifacts"][name]

    @ensure_load_before
    def check_integrity(self) -> dict[str, Any]:
        """Verify registry integrity against actual filesystem.

        Checks for:
        - Missing files: In registry but file doesn't exist
        - Extra files: Parquet file exists but not in registry
        - Row count mismatches: File row count differs from registry
        - Artifact missing files: Files in artifact entries that are missing
        - Artifact checksum mismatches: sha256 mismatch for artifact files

        Returns
        -------
        dict[str, Any]
            Dict with:
            - valid: bool - True if no issues found
            - missing_files: list[str] - Analyses in registry but file missing
            - extra_files: list[str] - Parquet files not in registry
            - row_count_mismatches: list[dict] - Analyses with row count mismatches
            - artifact_missing_files: list[dict] - Artifact missing file entries
            - artifact_checksum_mismatches: list[dict] - Artifact checksum mismatches

        """
        from loguru import logger

        assert self._registry is not None
        missing_files = []
        extra_files = []
        row_count_mismatches = []
        artifact_missing_files = []
        artifact_checksum_mismatches = []

        # Check for missing files and row count mismatches
        for name, entry in self._registry["analyses"].items():
            filename = entry["filename"]
            filepath = self.analysis_dir / filename

            if not filepath.exists():
                missing_files.append(name)
                logger.warning(f"Analysis '{name}' in registry but file missing: {filepath}")
            else:
                # Check row count
                try:
                    df = pd.read_parquet(filepath)
                    actual_rows = len(df)
                    expected_rows = entry["row_count"]

                    if actual_rows != expected_rows:
                        row_count_mismatches.append(
                            {
                                "name": name,
                                "expected": expected_rows,
                                "actual": actual_rows,
                            }
                        )
                        logger.warning(
                            f"Row count mismatch for '{name}': "
                            f"expected {expected_rows}, actual {actual_rows}"
                        )
                except Exception as e:
                    logger.warning(f"Error reading '{name}' for integrity check: {e}")

        # Check for extra parquet files not in registry
        if self.analysis_dir.exists():
            registered_files = {entry["filename"] for entry in self._registry["analyses"].values()}
            for parquet_file in self.analysis_dir.glob("*.parquet"):
                if parquet_file.name not in registered_files:
                    extra_files.append(parquet_file.name)
                    logger.warning(f"Orphaned parquet file not in registry: {parquet_file.name}")

        for name, entry in self._registry["artifacts"].items():
            for rel_path in entry.get("files", []):
                artifact_path = self.analysis_dir / rel_path
                if not artifact_path.exists():
                    artifact_missing_files.append({"artifact": name, "file": rel_path})
                    logger.warning(f"Artifact '{name}' file missing: {artifact_path}")
                    continue

                expected_checksum = entry.get("checksums", {}).get(rel_path)
                if expected_checksum:
                    actual_checksum = self._calculate_checksum(artifact_path)
                    if actual_checksum != expected_checksum:
                        artifact_checksum_mismatches.append(
                            {
                                "artifact": name,
                                "file": rel_path,
                                "expected": expected_checksum,
                                "actual": actual_checksum,
                            }
                        )
                        logger.warning(f"Artifact checksum mismatch for '{name}': {artifact_path}")

        valid = (
            len(missing_files) == 0
            and len(extra_files) == 0
            and len(row_count_mismatches) == 0
            and len(artifact_missing_files) == 0
            and len(artifact_checksum_mismatches) == 0
        )

        return {
            "valid": valid,
            "missing_files": missing_files,
            "extra_files": extra_files,
            "row_count_mismatches": row_count_mismatches,
            "artifact_missing_files": artifact_missing_files,
            "artifact_checksum_mismatches": artifact_checksum_mismatches,
        }

    def _create_default_registry(self) -> dict[str, Any]:
        """Create default empty registry structure."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "analyses": {},
            "artifacts": {},
        }

    def _ensure_registry_keys(self) -> None:
        """Ensure expected top-level keys exist in the registry."""
        if self._registry is None:
            return

        self._registry.setdefault("schema_version", self.SCHEMA_VERSION)
        self._registry.setdefault("analyses", {})
        self._registry.setdefault("artifacts", {})

    def _extract_metadata(self, df: pd.DataFrame) -> dict[str, Any]:
        """Extract metadata from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to extract metadata from.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``'row_count'`` and ``'columns'`` keys.

        """
        return {
            "row_count": len(df),
            "columns": list(df.columns),
        }

    def _get_timestamp(self) -> str:
        """Get ISO 8601 UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate sha256 checksum for a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @ensure_load_before
    @save_after
    def remove_entry(self, name: str) -> None:
        """Remove analysis entry from registry.

        Parameters
        ----------
        name : str
            Analysis name to remove

        Raises
        ------
        KeyError
            If analysis not found in registry

        """
        from loguru import logger

        assert self._registry is not None
        if name not in self._registry["analyses"]:
            raise KeyError(f"Analysis '{name}' not found in registry")

        del self._registry["analyses"][name]
        logger.debug(f"Removed analysis '{name}' from registry")
