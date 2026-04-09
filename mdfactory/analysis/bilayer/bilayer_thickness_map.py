# ABOUTME: Bilayer thickness mapping on an XY grid
# ABOUTME: Computes spatially resolved thickness from headgroup z-positions
"""Bilayer thickness mapping on an XY grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    lipid_species_by_resname,
    residue_mean_position,
    run_per_frame_analysis,
    trajectory_window,
)


def _bilayer_thickness_frame(
    atomgroup,
    species_map,
    grid_spacing,
) -> list[dict[str, float | int]]:
    universe = atomgroup.universe
    head_positions = []
    for resname, spec in species_map.items():
        residues = universe.select_atoms(f"resname {resname}").residues
        for residue in residues:
            pos = residue_mean_position(residue, spec["head_atoms"])
            if pos is not None:
                head_positions.append(pos)

    if not head_positions:
        return []

    head_positions = np.vstack(head_positions)
    x_len, y_len = universe.dimensions[:2]
    if x_len <= 0 or y_len <= 0:
        return []

    z_mid = float(np.median(head_positions[:, 2]))
    top_mask = head_positions[:, 2] > z_mid
    bottom_mask = head_positions[:, 2] <= z_mid

    n_bins_x = max(int(np.ceil(x_len / grid_spacing)), 1)
    n_bins_y = max(int(np.ceil(y_len / grid_spacing)), 1)
    x_edges = np.linspace(0, x_len, n_bins_x + 1)
    y_edges = np.linspace(0, y_len, n_bins_y + 1)

    def accumulate(mask):
        x_vals = np.mod(head_positions[:, 0], x_len)[mask]
        y_vals = np.mod(head_positions[:, 1], y_len)[mask]
        z_vals = head_positions[:, 2][mask]
        x_idx = np.clip(np.digitize(x_vals, x_edges) - 1, 0, n_bins_x - 1)
        y_idx = np.clip(np.digitize(y_vals, y_edges) - 1, 0, n_bins_y - 1)
        sums: dict[tuple[int, int], float] = {}
        counts: dict[tuple[int, int], int] = {}
        for i, j, z in zip(x_idx, y_idx, z_vals, strict=False):
            key = (int(i), int(j))
            sums[key] = sums.get(key, 0.0) + float(z)
            counts[key] = counts.get(key, 0) + 1
        return sums, counts

    time_ns = universe.trajectory.time / 1000.0
    frame_idx = universe.trajectory.frame
    rows: list[dict[str, float | int]] = []
    top_sums, top_counts = accumulate(top_mask)
    bottom_sums, bottom_counts = accumulate(bottom_mask)

    for x_idx in range(n_bins_x):
        for y_idx in range(n_bins_y):
            cell = (x_idx, y_idx)
            top_count = top_counts.get(cell, 0)
            bottom_count = bottom_counts.get(cell, 0)
            top_z = top_sums.get(cell, float("nan"))
            bottom_z = bottom_sums.get(cell, float("nan"))
            if top_count > 0:
                top_z = top_z / top_count
            if bottom_count > 0:
                bottom_z = bottom_z / bottom_count
            thickness = (
                float(top_z - bottom_z) if top_count > 0 and bottom_count > 0 else float("nan")
            )
            x_center = 0.5 * (x_edges[x_idx] + x_edges[x_idx + 1])
            y_center = 0.5 * (y_edges[y_idx] + y_edges[y_idx + 1])
            rows.append(
                {
                    "time_ns": time_ns,
                    "frame": frame_idx,
                    "x": float(x_center),
                    "y": float(y_center),
                    "thickness": thickness,
                    "top_z": float(top_z) if top_count > 0 else float("nan"),
                    "bottom_z": float(bottom_z) if bottom_count > 0 else float("nan"),
                    "top_count": int(top_count),
                    "bottom_count": int(bottom_count),
                }
            )
    return rows


def bilayer_thickness_map(
    simulation,
    *,
    grid_spacing: float = 20.0,
    start_ns: float | None = 100.0,
    stride: int = 1,
    backend: str = "multiprocessing",
    n_workers: int = 4,
) -> pd.DataFrame:
    """Compute bilayer thickness on an XY grid.

    Parameters
    ----------
    simulation : Simulation
        Simulation instance with universe and build input.
    grid_spacing : float
        Grid spacing in Angstrom.
    start_ns : float | None
        Start time for analysis in ns.
    stride : int
        Frame stride.
    backend : str
        MDAnalysis backend for parallel execution.
    n_workers : int
        Number of workers for parallel execution.

    Returns
    -------
    pd.DataFrame
        Columns: time_ns, frame, x, y, thickness, top_z, bottom_z, top_count, bottom_count.

    """
    u = simulation.universe
    lipid_species = lipid_species_by_resname(simulation.build_input)
    columns = [
        "time_ns",
        "frame",
        "x",
        "y",
        "thickness",
        "top_z",
        "bottom_z",
        "top_count",
        "bottom_count",
    ]
    if not lipid_species:
        return pd.DataFrame(columns=pd.Index(columns))

    start_frame, stop_frame, step = trajectory_window(u, start_ns=start_ns, stride=stride)

    species_map = {
        resname: {"head_atoms": spec.head_atoms} for resname, spec in lipid_species.items()
    }
    timeseries = run_per_frame_analysis(
        _bilayer_thickness_frame,
        u.trajectory,
        u.atoms,
        species_map,
        grid_spacing,
        start=start_frame,
        stop=stop_frame,
        step=step,
        backend=backend,
        n_workers=n_workers,
    )
    rows = [row for frame_rows in timeseries for row in frame_rows]
    return pd.DataFrame(rows)
