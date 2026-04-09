# ABOUTME: Lipid tail order parameter (SCD) analysis
# ABOUTME: Computes per-carbon deuterium order parameters along acyl chains
"""Lipid tail order parameter analysis."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .utils import lipid_species_by_resname, trajectory_window


def _tail_order_parameter_frame(
    _atomgroup,
    shared_tail_data,
    normal_vec,
) -> list[tuple[str, int, np.ndarray]]:
    frame_results: list[tuple[str, int, np.ndarray]] = []
    for resname, tails in shared_tail_data.items():
        for tail_idx, chains in enumerate(tails):
            if not chains:
                continue
            tail_length = len(chains[0][0])
            sums = np.zeros(tail_length)
            counts = np.zeros(tail_length)
            for carbons, hydrogens in chains:
                for idx, (carbon, h_list) in enumerate(zip(carbons, hydrogens, strict=False)):
                    s_val = _order_parameter(carbon, h_list, normal_vec)
                    sums[idx] += s_val
                    counts[idx] += 1
            averages = sums / np.maximum(counts, 1.0)
            frame_results.append((resname, tail_idx, averages))
    return frame_results


def _tail_order_parameter_frame_for_index(
    args: tuple[int, str, str, dict[str, list[list[tuple[list, list]]]], np.ndarray],
) -> list[tuple[str, int, np.ndarray]]:
    frame_idx, structure_path, trajectory_path, tail_data, normal_vec = args
    import MDAnalysis as mda

    u = mda.Universe(structure_path, trajectory_path)
    u.trajectory[frame_idx]
    return _tail_order_parameter_frame(u.atoms, tail_data, normal_vec)


def _trace_carbon_chain(residue, start_atom):
    chain = [start_atom]
    visited = {start_atom.index}
    if not hasattr(residue.atoms, "bonds") or len(residue.atoms.bonds) == 0:
        residue.atoms.guess_bonds()

    while True:
        current = chain[-1]
        next_carbon = None
        try:
            bonded = current.bonded_atoms
        except Exception:
            bonded = []

        for atom in bonded:
            if not atom.name.startswith("C"):
                continue
            if atom.index in visited or atom.residue != residue:
                continue
            try:
                n_h = len([a for a in atom.bonded_atoms if a.name.startswith("H")])
            except Exception:
                n_h = 0
            if n_h >= 1:
                next_carbon = atom
                break

        if next_carbon is None:
            break
        chain.append(next_carbon)
        visited.add(next_carbon.index)
        if len(chain) > 30:
            break

    return chain


def _bonded_hydrogens(carbon):
    try:
        return [atom for atom in carbon.bonded_atoms if atom.name.startswith("H")]
    except Exception:
        return []


def _order_parameter(carbon, hydrogens, normal):
    if not hydrogens:
        return 0.0
    s_values = []
    for hydrogen in hydrogens:
        vec = hydrogen.position - carbon.position
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        cos_val = np.dot(vec, normal) / norm
        s_values.append(1.5 * cos_val * cos_val - 0.5)
    if not s_values:
        return 0.0
    return float(np.mean(s_values))


def tail_order_parameter(
    simulation,
    *,
    stride: int = 5,
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    backend: str = "multiprocessing",
    n_workers: int = 4,
) -> pd.DataFrame:
    """Compute deuterium order parameters for lipid tails.

    Parameters
    ----------
    simulation : Simulation
        Simulation instance with universe and build input.
    stride : int
        Frame stride.
    normal : tuple[float, float, float]
        Membrane normal vector.
    backend : str
        MDAnalysis backend for parallel execution.
    n_workers : int
        Number of workers for parallel execution.

    Returns
    -------
    pd.DataFrame
        Columns: resname, tail_index, carbon_index, order_parameter.

    """
    u = simulation.universe
    u.atoms.guess_bonds()
    lipid_species = lipid_species_by_resname(simulation.build_input)
    if not lipid_species:
        return pd.DataFrame(
            columns=pd.Index(["resname", "tail_index", "carbon_index", "order_parameter"])
        )

    normal_vec = np.asarray(normal, dtype=float)
    normal_vec /= np.linalg.norm(normal_vec)

    tail_data: dict[str, list[list[tuple[list, list]]]] = {}
    for resname, spec in lipid_species.items():
        if not spec.tail_atoms:
            continue
        residues = u.select_atoms(f"resname {resname}").residues
        tails = []
        for tail_start_idx in spec.tail_atoms:
            tail_chains = []
            for residue in residues:
                if tail_start_idx >= len(residue.atoms):
                    continue
                start_atom = residue.atoms[tail_start_idx]
                if not start_atom.name.startswith("C"):
                    carbons = residue.atoms.select_atoms("name C*")
                    if len(carbons) == 0:
                        continue
                    start_atom = carbons[0]
                chain = _trace_carbon_chain(residue, start_atom)
                if not chain:
                    continue
                hydrogens = [_bonded_hydrogens(carbon) for carbon in chain]
                tail_chains.append((chain, hydrogens))
            tails.append(tail_chains)
        tail_data[resname] = tails

    start_frame, stop_frame, step = trajectory_window(u, stride=stride)

    frames = list(range(start_frame, stop_frame, step))
    frame_results = []
    if backend == "serial":
        for frame_idx in frames:
            u.trajectory[frame_idx]
            frame_results.append(_tail_order_parameter_frame(u.atoms, tail_data, normal_vec))
    else:
        from multiprocessing import Pool

        n_workers = max(n_workers, 1)

        args_list = [
            (
                frame_idx,
                str(simulation.structure_file),
                str(simulation.trajectory_file),
                tail_data,
                normal_vec,
            )
            for frame_idx in frames
        ]

        with Pool(processes=n_workers) as pool:
            frame_results = pool.map(_tail_order_parameter_frame_for_index, args_list)

    accumulators: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for frame_entries in frame_results:
        for resname, tail_idx, values in frame_entries:
            accumulators[(resname, int(tail_idx))].append(values)

    rows: list[dict[str, float | int | str]] = []
    for (resname, tail_idx), values_list in accumulators.items():
        if not values_list:
            continue
        mean_values = np.mean(np.vstack(values_list), axis=0)
        for carbon_idx, order_val in enumerate(mean_values, start=1):
            rows.append(
                {
                    "resname": resname,
                    "tail_index": tail_idx,
                    "carbon_index": carbon_idx,
                    "order_parameter": float(order_val),
                }
            )

    return pd.DataFrame(rows)
