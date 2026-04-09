# ABOUTME: Per-lipid radius of gyration analysis with XY/Z decomposition
# ABOUTME: Tracks Rg components over time to characterize lipid conformational dynamics
"""Per-lipid radius of gyration analysis with XY/Z decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import lipid_species_by_resname, run_per_frame_analysis, trajectory_window

DEFAULT_IL_RESNAMES = ["ILN", "ILP"]


def _compute_rg_components(positions: np.ndarray) -> tuple[float, float, float]:
    """Compute radius of gyration and its XY/Z components.

    Parameters
    ----------
    positions : np.ndarray
        Atom positions, shape (N, 3)

    Returns
    -------
    tuple[float, float, float]
        (rg_total, rg_xy, rg_z) in Angstroms

    """
    if len(positions) == 0:
        return 0.0, 0.0, 0.0

    center = positions.mean(axis=0)
    relative = positions - center

    # Rg_total = sqrt(mean(r^2))
    r_squared = np.sum(relative**2, axis=1)
    rg_total = np.sqrt(np.mean(r_squared))

    # Rg_xy = sqrt(mean(x^2 + y^2))
    xy_squared = relative[:, 0] ** 2 + relative[:, 1] ** 2
    rg_xy = np.sqrt(np.mean(xy_squared))

    # Rg_z = sqrt(mean(z^2))
    z_squared = relative[:, 2] ** 2
    rg_z = np.sqrt(np.mean(z_squared))

    return float(rg_total), float(rg_xy), float(rg_z)


def _lipid_rg_frame(
    atomgroup,
    residue_index_map: list[tuple[str, int, np.ndarray]],
    log_every: int | None = None,
) -> list[dict[str, float | int | str]]:
    """Compute per-lipid Rg components for a single frame.

    Parameters
    ----------
    atomgroup : AtomGroup
        MDAnalysis atom group (typically universe.atoms)
    residue_index_map : list[tuple[str, int, np.ndarray]]
        List of (resname, resid, atom_indices) tuples for residues to analyze.
    log_every : int | None
        Log every N frames when running in serial (None disables logging).

    Returns
    -------
    list[dict]
        List of dicts with per-lipid Rg data

    """
    universe = atomgroup.universe
    frame_idx = universe.trajectory.frame
    time_ns = universe.trajectory.time / 1000.0

    if log_every is not None and log_every > 0 and frame_idx % log_every == 0:
        from loguru import logger

        logger.info(f"lipid_rg frame {frame_idx} ({time_ns:.3f} ns)")

    rows: list[dict[str, float | int | str]] = []
    all_positions = atomgroup.positions

    for resname, resid, atom_indices in residue_index_map:
        positions = all_positions[atom_indices]
        rg_total, rg_xy, rg_z = _compute_rg_components(positions)
        rows.append(
            {
                "time_ns": time_ns,
                "frame": frame_idx,
                "resname": resname,
                "resid": resid,
                "rg_total": rg_total,
                "rg_xy": rg_xy,
                "rg_z": rg_z,
            }
        )

    return rows


def lipid_rg(
    simulation,
    *,
    species_filter: list[str] | None = None,
    start_ns: float | None = 0.0,
    stride: int = 1,
    backend: str = "multiprocessing",
    n_workers: int = 4,
    log_every: int | None = 50,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute per-lipid radius of gyration with XY/Z decomposition.

    Tracks Rg components for each lipid molecule at each frame to
    characterize conformational dynamics. By default analyzes IL
    (ionizable lipid) species.

    Parameters
    ----------
    simulation : Simulation
        Simulation instance with universe and build input.
    species_filter : list[str] | None
        Residue names to analyze. Default: ["ILN", "ILP"]
    start_ns : float | None
        Start time for analysis in ns.
    stride : int
        Frame stride.
    backend : str
        MDAnalysis backend for parallel execution.
    n_workers : int
        Number of workers for parallel execution.
    log_every : int | None
        Log every N frames when running in serial backend.
    verbose : bool
        Pass through to MDAnalysis AnalysisFromFunction.run to control progress output.

    Returns
    -------
    pd.DataFrame
        Columns: time_ns, frame, resname, resid, rg_total, rg_xy, rg_z

    """
    from loguru import logger

    u = simulation.universe

    # Determine target species
    if species_filter is None:
        # Default to IL species, but only include those present in simulation
        lipid_species = lipid_species_by_resname(simulation.build_input)
        target_resnames = [rn for rn in DEFAULT_IL_RESNAMES if rn in lipid_species]
    else:
        target_resnames = species_filter

    if not target_resnames:
        logger.warning("No target species found for lipid_rg analysis")
        return pd.DataFrame(
            columns=pd.Index(["time_ns", "frame", "resname", "resid", "rg_total", "rg_xy", "rg_z"])
        )

    start_frame, stop_frame, step = trajectory_window(u, start_ns=start_ns, stride=stride)
    n_frames = len(range(start_frame, stop_frame, step))

    logger.info(f"Running lipid_rg for species: {target_resnames}")
    logger.info(f"Frames {start_frame} to {stop_frame}, stride {step} ({n_frames} frames)")

    residue_index_map: list[tuple[str, int, np.ndarray]] = []
    for resname in target_resnames:
        selection = u.select_atoms(f"resname {resname}")
        if len(selection) == 0:
            continue
        for residue in selection.residues:
            residue_index_map.append((resname, int(residue.resid), residue.atoms.indices.copy()))

    if not residue_index_map:
        logger.warning("No residues found for lipid_rg analysis")
        return pd.DataFrame(
            columns=pd.Index(["time_ns", "frame", "resname", "resid", "rg_total", "rg_xy", "rg_z"])
        )

    # Avoid noisy multi-process logging by only emitting frame logs in serial.
    frame_log_every = log_every if backend == "serial" else None

    timeseries = run_per_frame_analysis(
        _lipid_rg_frame,
        u.trajectory,
        u.atoms,
        residue_index_map,
        frame_log_every,
        start=start_frame,
        stop=stop_frame,
        step=step,
        backend=backend,
        n_workers=n_workers,
        verbose=verbose,
    )

    rows = [row for frame_rows in timeseries for row in frame_rows]
    logger.info(f"Completed lipid_rg: {len(rows)} data points collected")
    return pd.DataFrame(rows)
