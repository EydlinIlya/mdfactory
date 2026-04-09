# ABOUTME: Lipid clustering and coordination number analysis
# ABOUTME: Computes nearest-neighbor species counts for each lipid type per leaflet
"""Lipid clustering and coordination analysis."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from MDAnalysis.analysis.distances import distance_array

from .utils import lipid_species_by_resname, run_per_frame_analysis, trajectory_window


def _lipid_clustering_frame(
    atomgroup,
    cutoff,
    residues,
    atom_to_lipid,
) -> list[dict[str, float | int | str]]:
    if len(atomgroup) == 0:
        return []

    universe = atomgroup.universe
    dist_matrix = distance_array(
        atomgroup.positions,
        atomgroup.positions,
        box=universe.dimensions,
    )
    within = np.where((dist_matrix <= cutoff) & (dist_matrix > 0))

    uf = _UnionFind(len(residues))
    neighbors = [set() for _ in range(len(residues))]

    for i, j in zip(within[0], within[1], strict=False):
        lipid_i = atom_to_lipid[int(i)]
        lipid_j = atom_to_lipid[int(j)]
        if lipid_i == lipid_j:
            continue
        uf.union(int(lipid_i), int(lipid_j))
        neighbors[int(lipid_i)].add(int(lipid_j))
        neighbors[int(lipid_j)].add(int(lipid_i))

    clusters = uf.clusters()
    cluster_sizes = {root: len(members) for root, members in clusters.items()}
    cluster_ids = {root: idx + 1 for idx, root in enumerate(clusters.keys())}
    frame_idx = universe.trajectory.frame
    time_ns = universe.trajectory.time / 1000.0
    rows: list[dict[str, float | int | str]] = []
    for idx, residue in enumerate(residues):
        root = uf.find(idx)
        rows.append(
            {
                "time_ns": time_ns,
                "frame": frame_idx,
                "resid": int(residue.resid),
                "resname": residue.resname,
                "cluster_id": cluster_ids[root],
                "cluster_size": cluster_sizes[root],
                "coordination_number": len(neighbors[idx]),
            }
        )
    return rows


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, idx: int) -> int:
        while self.parent[idx] != idx:
            self.parent[idx] = self.parent[self.parent[idx]]
            idx = self.parent[idx]
        return idx

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

    def clusters(self) -> dict[int, list[int]]:
        clusters: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(self.parent)):
            clusters[self.find(idx)].append(idx)
        return clusters


def lipid_clustering(
    simulation,
    *,
    cutoff: float = 4.0,
    stride: int = 1,
    backend: str = "multiprocessing",
    n_workers: int = 4,
) -> pd.DataFrame:
    """Cluster lipids based on atom-level contacts and compute coordination numbers.

    Parameters
    ----------
    simulation : Simulation
        Simulation instance with universe and build input.
    cutoff : float
        Distance cutoff in Angstrom.
    stride : int
        Frame stride.
    backend : str
        MDAnalysis backend for parallel execution.
    n_workers : int
        Number of workers for parallel execution.

    Returns
    -------
    pd.DataFrame
        Columns: time_ns, frame, resid, resname, cluster_id, cluster_size,
        coordination_number.

    """
    u = simulation.universe
    lipid_species = lipid_species_by_resname(simulation.build_input)
    columns = [
        "time_ns",
        "frame",
        "resid",
        "resname",
        "cluster_id",
        "cluster_size",
        "coordination_number",
    ]
    if not lipid_species:
        return pd.DataFrame(columns=pd.Index(columns))

    resnames = " ".join(lipid_species.keys())
    lipid_atoms = u.select_atoms(f"resname {resnames} and not name H*")
    residues = lipid_atoms.residues
    if len(residues) == 0:
        return pd.DataFrame(columns=pd.Index(columns))

    lipid_keys = [(res.resname, res.resid, res.segid) for res in residues]
    lipid_lookup = {key: idx for idx, key in enumerate(lipid_keys)}
    atom_to_lipid = np.zeros(len(lipid_atoms), dtype=int)
    for idx, atom in enumerate(lipid_atoms):
        atom_to_lipid[idx] = lipid_lookup[(atom.resname, atom.resid, atom.segid)]

    start_frame, stop_frame, step = trajectory_window(u, stride=stride)

    timeseries = run_per_frame_analysis(
        _lipid_clustering_frame,
        u.trajectory,
        lipid_atoms,
        cutoff,
        residues,
        atom_to_lipid,
        start=start_frame,
        stop=stop_frame,
        step=step,
        backend=backend,
        n_workers=n_workers,
    )
    records = [row for frame_rows in timeseries for row in frame_rows]
    return pd.DataFrame(records)
