# ABOUTME: End-to-end tail distance analysis for lipid species
# ABOUTME: Measures head-to-tail-tip vector length per lipid over trajectory frames
"""End-to-end tail distance analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import lipid_species_by_resname, residue_atoms, trajectory_window


def _tail_end_to_end_frame(
    atomgroup,
    species_map,
) -> list[tuple[str, int, np.ndarray]]:
    universe = atomgroup.universe
    frame_data: list[tuple[str, int, np.ndarray]] = []
    for resname, spec in species_map.items():
        if not spec["head_atoms"] or not spec["tail_atoms"]:
            continue
        residues = universe.select_atoms(f"resname {resname}").residues
        head_idx = spec["head_atoms"][0]
        for tail_idx, tail_atom_idx in enumerate(spec["tail_atoms"]):
            distances = []
            for residue in residues:
                head_atom = residue_atoms(residue, [head_idx])
                tail_atom = residue_atoms(residue, [tail_atom_idx])
                if head_atom is None or tail_atom is None:
                    continue
                dist = np.linalg.norm(head_atom.positions[0] - tail_atom.positions[0])
                distances.append(dist)
            if distances:
                frame_data.append((resname, tail_idx, np.array(distances)))
    return frame_data


def _tail_end_to_end_frame_for_index(
    args: tuple[int, str, str, dict[str, dict[str, list[int]]]],
) -> list[tuple[str, int, np.ndarray]]:
    frame_idx, structure_path, trajectory_path, species_map = args
    import MDAnalysis as mda

    u = mda.Universe(structure_path, trajectory_path)
    u.trajectory[frame_idx]
    return _tail_end_to_end_frame(u.atoms, species_map)


def tail_end_to_end(
    simulation,
    *,
    start_ns: float | None = 0.0,
    stride: int = 1,
    n_bins: int = 50,
    backend: str = "multiprocessing",
    n_workers: int = 4,
) -> pd.DataFrame:
    """Compute head-to-tail end-to-end distance distributions.

    Parameters
    ----------
    simulation : Simulation
        Simulation instance with universe and build input.
    start_ns : float | None
        Start time for analysis in ns.
    stride : int
        Frame stride.
    n_bins : int
        Number of histogram bins.
    backend : str
        MDAnalysis backend for parallel execution.
    n_workers : int
        Number of workers for parallel execution.

    Returns
    -------
    pd.DataFrame
        Columns: resname, tail_index, distance, probability_density.

    """
    u = simulation.universe
    lipid_species = lipid_species_by_resname(simulation.build_input)
    if not lipid_species:
        return pd.DataFrame(
            columns=pd.Index(["resname", "tail_index", "distance", "probability_density"])
        )

    start_frame, stop_frame, step = trajectory_window(u, start_ns=start_ns, stride=stride)

    species_map = {
        resname: {"head_atoms": spec.head_atoms, "tail_atoms": spec.tail_atoms}
        for resname, spec in lipid_species.items()
    }

    frames = list(range(start_frame, stop_frame, step))
    frame_results = []
    if backend == "serial":
        for frame_idx in frames:
            u.trajectory[frame_idx]
            frame_results.append(_tail_end_to_end_frame(u.atoms, species_map))
    else:
        from multiprocessing import Pool

        n_workers = max(n_workers, 1)

        args_list = [
            (
                frame_idx,
                str(simulation.structure_file),
                str(simulation.trajectory_file),
                species_map,
            )
            for frame_idx in frames
        ]

        with Pool(processes=n_workers) as pool:
            frame_results = pool.map(_tail_end_to_end_frame_for_index, args_list)

    distance_map: dict[tuple[str, int], list[float]] = {}
    for frame_entries in frame_results:
        for resname, tail_idx, distances in frame_entries:
            key = (resname, int(tail_idx))
            distance_map.setdefault(key, []).extend(distances.tolist())

    rows: list[dict[str, float | int | str]] = []
    for (resname, tail_idx), distances in distance_map.items():
        if not distances:
            continue
        hist, bin_edges = np.histogram(distances, bins=n_bins, density=True)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        for center, prob in zip(centers, hist, strict=False):
            rows.append(
                {
                    "resname": resname,
                    "tail_index": tail_idx,
                    "distance": float(center),
                    "probability_density": float(prob),
                }
            )

    return pd.DataFrame(rows)
