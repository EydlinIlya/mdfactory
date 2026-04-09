# ABOUTME: Area per lipid analysis using Voronoi tessellation of headgroup positions
# ABOUTME: Computes per-leaflet, per-species area metrics across trajectory frames
"""Area per lipid analysis using Voronoi tessellation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    lipid_species_by_resname,
    residue_mean_position,
    run_per_frame_analysis,
    trajectory_window,
)


def _area_per_lipid_frame(
    atomgroup,
    species_atoms_map,
) -> list[dict[str, float | int | str]]:
    try:
        import freud  # noqa: F401
    except ImportError:
        return []

    universe = atomgroup.universe
    positions = []
    labels = []
    for resname, head_atoms in species_atoms_map.items():
        residues = universe.select_atoms(f"resname {resname}").residues
        for residue in residues:
            pos = residue_mean_position(residue, head_atoms)
            if pos is not None:
                positions.append(pos)
                labels.append(resname)

    if not positions:
        return []

    positions_np = np.vstack(positions)
    z_mid = float(np.median(positions_np[:, 2]))
    rows: list[dict[str, float | int | str]] = []
    frame_idx = universe.trajectory.frame
    time_ns = universe.trajectory.time / 1000.0

    for leaflet, mask in (
        ("upper", positions_np[:, 2] > z_mid),
        ("lower", positions_np[:, 2] <= z_mid),
    ):
        if not np.any(mask):
            continue
        leaflet_positions = positions_np[mask][:, :2]
        leaflet_labels = [lab for lab, keep in zip(labels, mask, strict=False) if keep]

        if len(leaflet_positions) < 3:
            continue

        box = freud.box.Box(Lx=universe.dimensions[0], Ly=universe.dimensions[1], is2D=True)
        voronoi = freud.locality.Voronoi()
        positions_wrapped = leaflet_positions.copy()
        positions_wrapped[:, 0] = positions_wrapped[:, 0] % box.Lx
        positions_wrapped[:, 1] = positions_wrapped[:, 1] % box.Ly
        _, inverse, counts = np.unique(
            positions_wrapped, axis=0, return_inverse=True, return_counts=True
        )
        if np.any(counts > 1):
            eps = 1e-3
            for idx, count in enumerate(counts):
                if count <= 1:
                    continue
                dup_indices = np.where(inverse == idx)[0]
                offsets = np.linspace(-eps, eps, num=count)
                positions_wrapped[dup_indices, 0] += offsets
                positions_wrapped[dup_indices, 1] += offsets[::-1]
            positions_wrapped[:, 0] = positions_wrapped[:, 0] % box.Lx
            positions_wrapped[:, 1] = positions_wrapped[:, 1] % box.Ly

        points = np.column_stack((positions_wrapped, np.zeros(len(positions_wrapped), dtype=float)))
        voronoi.compute((box, points))
        areas = voronoi.volumes

        rows.append(
            {
                "time_ns": time_ns,
                "frame": frame_idx,
                "leaflet": leaflet,
                "species": "all",
                "apl": float(np.mean(areas)),
                "n_lipids": int(len(areas)),
            }
        )

        for resname in sorted(set(leaflet_labels)):
            species_mask = np.array([lab == resname for lab in leaflet_labels])
            species_areas = areas[species_mask]
            if len(species_areas) == 0:
                continue
            rows.append(
                {
                    "time_ns": time_ns,
                    "frame": frame_idx,
                    "leaflet": leaflet,
                    "species": resname,
                    "apl": float(np.mean(species_areas)),
                    "n_lipids": int(len(species_areas)),
                }
            )
    return rows


def area_per_lipid(
    simulation,
    *,
    start_ns: float | None = None,
    last_ns: float | None = 300.0,
    stride: int = 1,
    backend: str = "multiprocessing",
    n_workers: int = 4,
) -> pd.DataFrame:
    """Compute area per lipid from headgroup XY Voronoi tessellation.

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
        Area per lipid data with columns time_ns, frame, leaflet, species, apl, n_lipids.

    """
    try:
        import freud  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "freud-analysis is required for area_per_lipid. Install freud-analysis."
        ) from exc

    u = simulation.universe
    lipid_species = lipid_species_by_resname(simulation.build_input)
    empty_columns = pd.Index(["time_ns", "frame", "leaflet", "species", "apl", "n_lipids"])
    if not lipid_species:
        return pd.DataFrame(columns=empty_columns)

    species_atoms = {resname: spec.head_atoms for resname, spec in lipid_species.items()}
    start_frame, stop_frame, step = trajectory_window(
        u, start_ns=start_ns, last_ns=last_ns, stride=stride
    )

    timeseries = run_per_frame_analysis(
        _area_per_lipid_frame,
        u.trajectory,
        u.atoms,
        species_atoms,
        start=start_frame,
        stop=stop_frame,
        step=step,
        backend=backend,
        n_workers=n_workers,
    )
    rows = [row for frame_rows in timeseries for row in frame_rows]
    return pd.DataFrame(rows)
