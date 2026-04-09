# ABOUTME: Computes z-axis density distributions for lipid head/tail atoms and water.
# ABOUTME: Profiles are centered on the membrane center of mass each frame.

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    lipid_species_by_resname,
    run_per_frame_analysis,
    tail_positions,
    trajectory_window,
    water_resnames,
)


def _membrane_center_z(universe, lipid_resnames: list[str]) -> float:
    """Calculate center of mass of all lipid atoms along z-axis."""
    if not lipid_resnames:
        return 0.0
    lipid_sel = universe.select_atoms("resname " + " ".join(lipid_resnames))
    if len(lipid_sel) == 0:
        return 0.0
    return float(lipid_sel.center_of_mass()[2])


def _density_distribution_frame(
    atomgroup,
    lipid_map,
    water_list,
    bins,
    z_centers,
) -> list[dict[str, float | int | str]]:
    universe = atomgroup.universe
    frame_idx = universe.trajectory.frame
    time_ns = universe.trajectory.time / 1000.0
    rows: list[dict[str, float | int | str]] = []

    membrane_center_z = _membrane_center_z(universe, list(lipid_map.keys()))

    for resname, spec in lipid_map.items():
        head_positions = tail_positions(universe, resname, spec["head_atoms"])
        tail_pos = tail_positions(universe, resname, spec["tail_atoms"])

        for group, positions in (("head", head_positions), ("tail", tail_pos)):
            z_vals = positions[:, 2] - membrane_center_z if positions.size else np.array([])
            counts, _ = np.histogram(z_vals, bins=bins)
            total = counts.sum()
            density = counts / total * 100.0 if total > 0 else np.zeros_like(counts, dtype=float)
            for z, d in zip(z_centers, density, strict=False):
                rows.append(
                    {
                        "time_ns": time_ns,
                        "frame": frame_idx,
                        "z": float(z),
                        "density_percent": float(d),
                        "species": resname,
                        "group": group,
                    }
                )

    if water_list:
        water_sel = universe.select_atoms("resname " + " ".join(water_list))
        z_vals = (
            water_sel.positions[:, 2] - membrane_center_z if len(water_sel) > 0 else np.array([])
        )
        counts, _ = np.histogram(z_vals, bins=bins)
        total = counts.sum()
        density = counts / total * 100.0 if total > 0 else np.zeros_like(counts, dtype=float)
        for z, d in zip(z_centers, density, strict=False):
            rows.append(
                {
                    "time_ns": time_ns,
                    "frame": frame_idx,
                    "z": float(z),
                    "density_percent": float(d),
                    "species": "water",
                    "group": "water",
                }
            )

    return rows


def density_distribution(
    simulation,
    *,
    start_ns: float | None = 0.0,
    end_ns: float | None = None,
    last_ns: float | None = None,
    stride: int = 1,
    bin_width: float = 1.0,
    backend: str = "multiprocessing",
    n_workers: int = 4,
) -> pd.DataFrame:
    """
    Compute z-density distributions for lipid head/tail atoms and water.

    Parameters
    ----------
    simulation : Simulation
        Simulation instance with universe and build input.
    start_ns : float | None
        Start time for analysis (ns). Use None to start at frame 0.
        Ignored when *last_ns* is set.
    end_ns : float | None
        End time for analysis (ns). Use None for end of trajectory.
    last_ns : float | None
        Analyze the last N ns of the trajectory. Overrides *start_ns*.
    stride : int
        Frame stride.
    bin_width : float
        Bin width in Angstrom.
    backend : str
        MDAnalysis backend for parallel execution.
    n_workers : int
        Number of workers for parallel execution.

    Returns
    -------
    pd.DataFrame
        Columns: time_ns, frame, z, density_percent, species, group.
    """
    u = simulation.universe
    lipid_species = lipid_species_by_resname(simulation.build_input)
    water_names = water_resnames(simulation.build_input)
    start_frame, stop_frame, step = trajectory_window(
        u, start_ns=start_ns, end_ns=end_ns, last_ns=last_ns, stride=stride
    )

    if len(u.atoms) == 0:
        return pd.DataFrame(
            columns=pd.Index(["time_ns", "frame", "z", "density_percent", "species", "group"])
        )

    u.trajectory[start_frame]
    z_extent = float(u.dimensions[2] / 2.0)
    bins = np.arange(-z_extent, z_extent + bin_width, bin_width)
    z_centers = 0.5 * (bins[:-1] + bins[1:])

    lipid_map = {
        resname: {"head_atoms": spec.head_atoms, "tail_atoms": spec.tail_atoms}
        for resname, spec in lipid_species.items()
    }
    timeseries = run_per_frame_analysis(
        _density_distribution_frame,
        u.trajectory,
        u.atoms,
        lipid_map,
        water_names,
        bins,
        z_centers,
        start=start_frame,
        stop=stop_frame,
        step=step,
        backend=backend,
        n_workers=n_workers,
    )
    rows = [row for frame_rows in timeseries for row in frame_rows]
    return pd.DataFrame(rows)
