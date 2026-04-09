# ABOUTME: Cholesterol tilt angle analysis relative to the bilayer normal
# ABOUTME: Measures per-leaflet cholesterol orientation over trajectory frames
"""Cholesterol tilt angle analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    lipid_species_by_resname,
    residue_mean_position,
    run_per_frame_analysis,
    trajectory_window,
)


def _cholesterol_tilt_frame(
    atomgroup,
    species_map,
    z_axis,
) -> list[dict[str, float | int | str]]:
    universe = atomgroup.universe
    head_positions = []
    per_residue = []
    for resname, spec in species_map.items():
        residues = universe.select_atoms(f"resname {resname}").residues
        for residue in residues:
            head_pos = residue_mean_position(residue, spec["head_atoms"])
            tail_pos = residue_mean_position(residue, spec["tail_atoms"])
            if head_pos is None or tail_pos is None:
                continue
            vector = head_pos - tail_pos
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue
            head_positions.append(head_pos)
            per_residue.append((residue.resid, resname, head_pos[2], vector / norm))

    if not per_residue:
        return []

    midplane = float(np.mean([pos[2] for pos in head_positions]))
    frame_idx = universe.trajectory.frame
    time_ns = universe.trajectory.time / 1000.0
    rows: list[dict[str, float | int | str]] = []
    for resid, resname, head_z, unit_vec in per_residue:
        leaflet = "upper" if head_z >= midplane else "lower"
        normal = z_axis if leaflet == "upper" else -z_axis
        cosang = np.dot(unit_vec, normal)
        angle = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        rows.append(
            {
                "time_ns": time_ns,
                "frame": frame_idx,
                "resid": int(resid),
                "resname": resname,
                "leaflet": leaflet,
                "tilt_deg": float(angle),
            }
        )
    return rows


def cholesterol_tilt(
    simulation,
    *,
    start_ns: float | None = None,
    last_ns: float | None = 200.0,
    stride: int = 1,
    backend: str = "multiprocessing",
    n_workers: int = 4,
) -> pd.DataFrame:
    """Compute cholesterol tilt angles relative to the bilayer normal.

    Parameters
    ----------
    simulation : Simulation
        Simulation instance with universe and build input.
    start_ns : float | None
        Start analysis from this time in nanoseconds. Ignored when *last_ns* is set.
    last_ns : float | None
        Analyze the last N ns of the trajectory. Use None for full trajectory.
    stride : int
        Frame stride.
    backend : str
        MDAnalysis backend for parallel execution.
    n_workers : int
        Number of workers for parallel execution.

    Returns
    -------
    pd.DataFrame
        Columns: time_ns, frame, resid, resname, leaflet, tilt_deg.

    """
    u = simulation.universe
    lipid_species = lipid_species_by_resname(simulation.build_input)
    empty_columns = pd.Index(["time_ns", "frame", "resid", "resname", "leaflet", "tilt_deg"])
    chol_species = {name: spec for name, spec in lipid_species.items() if name in {"CHL", "CHOL"}}
    if not chol_species:
        return pd.DataFrame(columns=empty_columns)

    start_frame, stop_frame, step = trajectory_window(
        u, start_ns=start_ns, last_ns=last_ns, stride=stride
    )
    z_axis = np.array([0.0, 0.0, 1.0])

    species_map = {
        resname: {"head_atoms": spec.head_atoms, "tail_atoms": spec.tail_atoms}
        for resname, spec in chol_species.items()
    }
    timeseries = run_per_frame_analysis(
        _cholesterol_tilt_frame,
        u.trajectory,
        u.atoms,
        species_map,
        z_axis,
        start=start_frame,
        stop=stop_frame,
        step=step,
        backend=backend,
        n_workers=n_workers,
    )
    records = [row for frame_rows in timeseries for row in frame_rows]
    return pd.DataFrame(records)
