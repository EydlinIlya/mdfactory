# ABOUTME: Helper functions for pushing artifact metadata to database
# ABOUTME: Discovers artifacts from simulations and uploads metadata to artifact tables

from pathlib import Path
from typing import Any

from loguru import logger

from ..analysis.simulation import Simulation
from .push import discover_simulation_folders
from .push_analysis import (
    get_artifact_table_name,
    get_artifacts_for_simulation_type,
    prepare_artifact_record,
    prepare_overview_record,
    update_overview_records,
    upload_analysis_data,
)


def discover_and_prepare_artifact_data(
    simulations: list[tuple[Path, Any]],
    artifact_name: str | None = None,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Discover artifact data from simulation folders.

    Parameters
    ----------
    simulations : list[tuple[Path, BuildInput]]
        List of (folder_path, build_input) tuples
    artifact_name : str | None
        Specific artifact to discover, or None for all

    Returns
    -------
    tuple[dict[str, list[dict]], list[dict]]
        (artifact_records_by_table, overview_records)
        artifact_records_by_table maps table_name to list of records
    """
    artifact_records: dict[str, list[dict]] = {}
    overview_records: list[dict] = []

    for folder, build_input in simulations:
        try:
            sim = Simulation(folder, build_input=build_input, trajectory_file=None)
        except Exception as e:
            logger.warning(f"Failed to create Simulation for {folder}: {e}")
            continue

        sim_type = sim.build_input.simulation_type

        # Get artifacts to check
        if artifact_name:
            artifacts_to_check = [artifact_name]
        else:
            artifacts_to_check = get_artifacts_for_simulation_type(sim_type)

        # Process artifacts
        completed_artifacts = set(sim.list_artifacts())
        for name in artifacts_to_check:
            if name in completed_artifacts:
                # Completed artifact - prepare full record
                record = prepare_artifact_record(sim, name)
                if record:
                    table_name = get_artifact_table_name(name)
                    if table_name not in artifact_records:
                        artifact_records[table_name] = []
                    artifact_records[table_name].append(record)

                    # Add completed overview entry
                    overview_records.append(
                        prepare_overview_record(
                            sim,
                            "artifact",
                            name,
                            "completed",
                            file_count=record["file_count"],
                        )
                    )
            else:
                # Not completed - add not_yet_run overview entry
                overview_records.append(
                    prepare_overview_record(sim, "artifact", name, "not_yet_run")
                )

    return artifact_records, overview_records


def push_artifacts(
    source: Path | None = None,
    csv: Path | None = None,
    csv_root: Path | None = None,
    artifact_name: str | None = None,
    force: bool = False,
    diff: bool = False,
) -> dict[str, int]:
    """Push artifact metadata to database.

    Parameters
    ----------
    source : Path | None
        Directory path, glob pattern, or summary YAML file (auto-detected)
    csv : Path | None
        CSV file with build specifications
    csv_root : Path | None
        Root directory for CSV hash search
    artifact_name : str | None
        Specific artifact to push, or None for all
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
    artifact_records, overview_records = discover_and_prepare_artifact_data(
        simulations,
        artifact_name=artifact_name,
    )

    results = {}

    # Upload artifact data
    for table_name, table_records in artifact_records.items():
        count = upload_analysis_data(table_records, table_name, force=force, diff=diff)
        results[table_name] = count

    # Update overview table
    if overview_records:
        count = update_overview_records(overview_records, force=force, diff=diff)
        results["ANALYSIS_OVERVIEW"] = count

    return results
