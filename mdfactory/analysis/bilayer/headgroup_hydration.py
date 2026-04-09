# ABOUTME: Headgroup hydration analysis using radial distribution functions
# ABOUTME: Computes water-headgroup RDFs and coordination numbers per lipid species
"""Headgroup hydration analysis using RDFs."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.rdf import InterRDF

from .utils import lipid_species_by_resname, trajectory_window, water_resnames


def _volume_frame(atomgroup, *_args) -> float:
    dims = atomgroup.universe.dimensions
    return float(dims[0] * dims[1] * dims[2])


def _first_minimum(r: np.ndarray, g_r: np.ndarray) -> float:
    if len(r) == 0:
        return 0.0
    peak_idx = int(np.argmax(g_r))
    if peak_idx >= len(r) - 1:
        return float(r[-1])
    min_idx = peak_idx + int(np.argmin(g_r[peak_idx:]))
    return float(r[min_idx])


def headgroup_hydration(
    simulation,
    *,
    start_ns: float | None = 100.0,
    stride: int = 5,
    r_max: float = 10.0,
    n_bins: int = 200,
    backend: str = "multiprocessing",
    n_workers: int = 4,
) -> pd.DataFrame:
    """Compute headgroup-water RDFs and hydration numbers.

    Parameters
    ----------
    simulation : Simulation
        Simulation instance with universe and build input.
    start_ns : float | None
        Start time for analysis in ns.
    stride : int
        Frame stride.
    r_max : float
        Maximum distance for RDF in Angstrom.
    n_bins : int
        Number of RDF bins.
    backend : str
        MDAnalysis backend for parallel execution.
    n_workers : int
        Number of workers for parallel execution.

    Returns
    -------
    pd.DataFrame
        Columns: resname, r, g_r, hydration_number, first_min_r.

    """
    u = simulation.universe
    lipid_species = lipid_species_by_resname(simulation.build_input)
    water_names = water_resnames(simulation.build_input)
    water_selection = "resname " + " ".join(water_names) if water_names else "resname SOL WAT HOH"
    water = u.select_atoms(water_selection)
    if len(water) == 0:
        return pd.DataFrame(
            columns=pd.Index(["resname", "r", "g_r", "hydration_number", "first_min_r"])
        )

    start_frame, stop_frame, step = trajectory_window(u, start_ns=start_ns, stride=stride)

    volume_analysis = AnalysisFromFunction(_volume_frame, u.trajectory, u.atoms)
    volume_analysis.run(
        start=start_frame,
        stop=stop_frame,
        step=step,
        backend=backend,
        n_workers=n_workers,
    )
    volumes = np.array(volume_analysis.results.timeseries)
    if volumes.size == 0:
        return pd.DataFrame(
            columns=pd.Index(["resname", "r", "g_r", "hydration_number", "first_min_r"])
        )

    number_density = len(water.residues) / float(np.mean(volumes))
    rows: list[dict[str, float | str]] = []

    for resname, spec in lipid_species.items():
        residues = u.select_atoms(f"resname {resname}").residues
        if len(residues) == 0 or not spec.head_atoms:
            continue
        head_indices = []
        for residue in residues:
            head_indices.extend(residue.atoms[spec.head_atoms].indices)
        if not head_indices:
            continue
        headgroup = u.atoms[head_indices]

        rdf = InterRDF(headgroup, water, nbins=n_bins, range=(0.0, r_max))
        try:
            rdf.run(
                start=start_frame,
                stop=stop_frame,
                step=step,
                backend=backend,
                n_workers=n_workers,
            )
        except TypeError:
            warnings.warn(
                "InterRDF does not support parallel backends; running serially.",
                RuntimeWarning,
                stacklevel=2,
            )
            rdf.run(start=start_frame, stop=stop_frame, step=step)
        r = rdf.bins
        g_r = rdf.rdf

        first_min_r = _first_minimum(r, g_r)
        mask = r <= first_min_r
        hydration = 4.0 * np.pi * number_density * np.trapz(g_r[mask] * r[mask] ** 2, r[mask])

        for radius, g_val in zip(r, g_r, strict=False):
            rows.append(
                {
                    "resname": resname,
                    "r": float(radius),
                    "g_r": float(g_val),
                    "hydration_number": float(hydration),
                    "first_min_r": float(first_min_r),
                }
            )

    return pd.DataFrame(rows)
