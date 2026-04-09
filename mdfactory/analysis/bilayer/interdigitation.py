# ABOUTME: Lipid tail interdigitation analysis between bilayer leaflets
# ABOUTME: Measures overlap of tail atom z-distributions across the midplane
"""Lipid tail interdigitation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    lipid_species_by_resname,
    residue_mean_position,
    run_per_frame_analysis,
    trajectory_window,
)


def _interdigitation_frame(
    atomgroup,
    species_map,
    bin_width,
) -> dict[str, float]:
    universe = atomgroup.universe
    head_positions = []
    residue_info = []
    for resname, spec in species_map.items():
        residues = universe.select_atoms(f"resname {resname}").residues
        for residue in residues:
            head_pos = residue_mean_position(residue, spec["head_atoms"])
            if head_pos is None:
                continue
            head_positions.append(head_pos)
            residue_info.append((resname, residue, head_pos[2], spec["tail_atoms"]))

    if not residue_info:
        return {}

    z_center = float(np.median([pos[2] for pos in head_positions]))
    upper_tails = []
    lower_tails = []
    penetration = {resname: {"upper": [], "lower": []} for resname in species_map}

    for resname, residue, head_z, tail_atoms in residue_info:
        if not tail_atoms:
            continue
        tail_z = residue.atoms[tail_atoms].positions[:, 2]
        if head_z > z_center:
            upper_tails.extend(tail_z)
            penetration[resname]["upper"].append(max(0.0, z_center - float(np.min(tail_z))))
        else:
            lower_tails.extend(tail_z)
            penetration[resname]["lower"].append(max(0.0, float(np.max(tail_z)) - z_center))

    if not upper_tails or not lower_tails:
        return {}

    z_max = universe.dimensions[2] / 2.0
    bins = np.arange(-z_max, z_max + bin_width, bin_width)
    hist_upper, _ = np.histogram(upper_tails, bins=bins)
    hist_lower, _ = np.histogram(lower_tails, bins=bins)
    density_upper = hist_upper / (len(upper_tails) * bin_width)
    density_lower = hist_lower / (len(lower_tails) * bin_width)
    overlap = np.minimum(density_upper, density_lower)
    overlap_integral = np.sum(overlap) * bin_width
    upper_integral = np.sum(density_upper) * bin_width
    lower_integral = np.sum(density_lower) * bin_width
    if upper_integral > 0 and lower_integral > 0:
        interdig_index = overlap_integral / min(upper_integral, lower_integral)
    else:
        interdig_index = 0.0

    row = {
        "time_ns": universe.trajectory.time / 1000.0,
        "frame": universe.trajectory.frame,
        "interdigitation_index": float(interdig_index),
        "z_center": float(z_center),
        "overlap_integral": float(overlap_integral),
        "upper_integral": float(upper_integral),
        "lower_integral": float(lower_integral),
    }
    for resname in species_map:
        upper_vals = penetration[resname]["upper"]
        lower_vals = penetration[resname]["lower"]
        all_vals = upper_vals + lower_vals
        row[f"penetration_{resname}"] = float(np.mean(all_vals)) if all_vals else np.nan
        row[f"penetration_{resname}_upper"] = float(np.mean(upper_vals)) if upper_vals else np.nan
        row[f"penetration_{resname}_lower"] = float(np.mean(lower_vals)) if lower_vals else np.nan
    return row


def interdigitation(
    simulation,
    *,
    last_ns: float | None = 300.0,
    stride: int = 1,
    bin_width: float = 0.5,
    backend: str = "multiprocessing",
    n_workers: int = 4,
) -> pd.DataFrame:
    """Compute interdigitation index and penetration depths per frame.

    Parameters
    ----------
    simulation : Simulation
        Simulation instance with universe and build input.
    last_ns : float | None
        Analyze the last N ns of the trajectory. Use None for full trajectory.
    stride : int
        Frame stride.
    bin_width : float
        Bin width for density profiles in Angstrom.
    backend : str
        MDAnalysis backend for parallel execution.
    n_workers : int
        Number of workers for parallel execution.

    Returns
    -------
    pd.DataFrame
        Time series with interdigitation index and per-species penetration depths.

    """
    u = simulation.universe
    lipid_species = lipid_species_by_resname(simulation.build_input)
    if not lipid_species:
        return pd.DataFrame(columns=pd.Index(["time_ns", "frame", "interdigitation_index"]))

    start_frame, stop_frame, step = trajectory_window(u, last_ns=last_ns, stride=stride)

    species_map = {
        resname: {"head_atoms": spec.head_atoms, "tail_atoms": spec.tail_atoms}
        for resname, spec in lipid_species.items()
    }
    timeseries = run_per_frame_analysis(
        _interdigitation_frame,
        u.trajectory,
        u.atoms,
        species_map,
        bin_width,
        start=start_frame,
        stop=stop_frame,
        step=step,
        backend=backend,
        n_workers=n_workers,
    )
    rows = [row for row in timeseries if row]
    return pd.DataFrame(rows)
