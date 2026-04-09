# ABOUTME: Registry query functions for analysis and artifact lookups
# ABOUTME: Provides utilities to query registered analyses/artifacts by type

from .artifacts import ARTIFACT_REGISTRY
from .simulation import ANALYSIS_REGISTRY


def get_analysis_table_name(analysis_name: str) -> str:
    """Convert analysis_name to table name.

    Parameters
    ----------
    analysis_name : str
        Analysis name (e.g., 'area_per_lipid')

    Returns
    -------
    str
        Table name (e.g., 'ANALYSIS_AREA_PER_LIPID')
    """
    return f"ANALYSIS_{analysis_name.upper()}"


def get_artifact_table_name(artifact_name: str) -> str:
    """Convert artifact_name to table name.

    Parameters
    ----------
    artifact_name : str
        Artifact name (e.g., 'bilayer_snapshot')

    Returns
    -------
    str
        Table name (e.g., 'ARTIFACT_BILAYER_SNAPSHOT')
    """
    return f"ARTIFACT_{artifact_name.upper()}"


def get_all_analysis_names() -> list[str]:
    """Get all registered analysis names across all simulation types.

    Returns
    -------
    list[str]
        Sorted list of unique analysis names
    """
    all_names = set()
    for sim_type_analyses in ANALYSIS_REGISTRY.values():
        all_names.update(sim_type_analyses.keys())
    return sorted(all_names)


def get_all_artifact_names() -> list[str]:
    """Get all registered artifact names across all simulation types.

    Returns
    -------
    list[str]
        Sorted list of unique artifact names
    """
    all_names = set()
    for sim_type_artifacts in ARTIFACT_REGISTRY.values():
        all_names.update(sim_type_artifacts.keys())
    return sorted(all_names)


def get_analyses_for_simulation_type(simulation_type: str) -> list[str]:
    """Get analysis names available for a simulation type.

    Parameters
    ----------
    simulation_type : str
        Simulation type (e.g., 'bilayer', 'mixedbox')

    Returns
    -------
    list[str]
        Sorted list of analysis names for this type
    """
    if simulation_type not in ANALYSIS_REGISTRY:
        return []
    return sorted(ANALYSIS_REGISTRY[simulation_type].keys())


def get_artifacts_for_simulation_type(simulation_type: str) -> list[str]:
    """Get artifact names available for a simulation type.

    Parameters
    ----------
    simulation_type : str
        Simulation type (e.g., 'bilayer', 'mixedbox')

    Returns
    -------
    list[str]
        Sorted list of artifact names for this type
    """
    if simulation_type not in ARTIFACT_REGISTRY:
        return []
    return sorted(ARTIFACT_REGISTRY[simulation_type].keys())
