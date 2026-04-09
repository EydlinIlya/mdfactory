# ABOUTME: Leaflet distribution analysis classifying lipids as top, mid, or bottom
# ABOUTME: Tracks per-species leaflet populations over trajectory frames
"""Leaflet/top-mid-bottom lipid distribution analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    lipid_species_by_resname,
    residue_mean_position,
    run_per_frame_analysis,
    trajectory_window,
)


def _leaflet_distribution_frame(
    atomgroup,
    species_map,
    z1_value,
) -> list[dict[str, float | int | str]]:
    universe = atomgroup.universe
    all_lipids = universe.select_atoms("resname " + " ".join(species_map.keys()))
    if len(all_lipids) == 0:
        return []
    z_coords = all_lipids.positions[:, 2]
    z_min = float(np.min(z_coords))
    z_max = float(np.max(z_coords))
    bilayer_thickness = z_max - z_min
    z1_height = bilayer_thickness * z1_value
    bottom_max = z_min + z1_height
    top_min = z_max - z1_height

    frame_idx = universe.trajectory.frame
    time_ns = universe.trajectory.time / 1000.0
    rows: list[dict[str, float | int | str]] = []

    for resname, spec in species_map.items():
        residues = universe.select_atoms(f"resname {resname}").residues
        counts = {"top": 0, "mid": 0, "bottom": 0}
        for residue in residues:
            head_pos = residue_mean_position(residue, spec["head_atoms"])
            if head_pos is None:
                continue
            z_val = head_pos[2]
            if z_val <= bottom_max:
                counts["bottom"] += 1
            elif z_val >= top_min:
                counts["top"] += 1
            else:
                counts["mid"] += 1

        for region, count in counts.items():
            rows.append(
                {
                    "time_ns": time_ns,
                    "frame": frame_idx,
                    "resname": resname,
                    "region": region,
                    "count": int(count),
                }
            )
    return rows


def leaflet_distribution(
    simulation,
    *,
    z1_fraction: float = 0.25,
    stride: int = 1,
    backend: str = "multiprocessing",
    n_workers: int = 4,
) -> pd.DataFrame:
    """Count lipid species in top, mid, bottom regions over time.

    Parameters
    ----------
    simulation : Simulation
        Simulation instance with universe and build input.
    z1_fraction : float
        Fraction of bilayer thickness assigned to top/bottom regions.
    stride : int
        Frame stride.
    backend : str
        MDAnalysis backend for parallel execution.
    n_workers : int
        Number of workers for parallel execution.

    Returns
    -------
    pd.DataFrame
        Columns: time_ns, frame, resname, region, count.

    """
    u = simulation.universe
    lipid_species = lipid_species_by_resname(simulation.build_input)
    if not lipid_species:
        return pd.DataFrame(columns=pd.Index(["time_ns", "frame", "resname", "region", "count"]))

    start_frame, stop_frame, step = trajectory_window(u, stride=stride)

    species_map = {
        resname: {"head_atoms": spec.head_atoms} for resname, spec in lipid_species.items()
    }
    timeseries = run_per_frame_analysis(
        _leaflet_distribution_frame,
        u.trajectory,
        u.atoms,
        species_map,
        z1_fraction,
        start=start_frame,
        stop=stop_frame,
        step=step,
        backend=backend,
        n_workers=n_workers,
    )
    rows = [row for frame_rows in timeseries for row in frame_rows]
    return pd.DataFrame(rows)
