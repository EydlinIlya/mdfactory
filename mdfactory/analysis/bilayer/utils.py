# ABOUTME: Shared utilities for bilayer analyses
# ABOUTME: Provides trajectory windowing, per-frame analysis, and lipid selection helpers
"""Shared utilities for bilayer analyses."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from MDAnalysis.analysis.base import AnalysisFromFunction

from mdfactory.models.species import LipidSpecies

if TYPE_CHECKING:
    import MDAnalysis as mda
    from MDAnalysis.core.groups import Residue

    from mdfactory.models.input import BuildInput


def species_by_resname(build_input: "BuildInput") -> dict[str, object]:
    """Return a mapping of residue name to species object."""
    return {spec.resname: spec for spec in build_input.system.species}


def lipid_species_by_resname(build_input: "BuildInput") -> dict[str, LipidSpecies]:
    """Return lipid species keyed by residue name."""
    return {
        spec.resname: spec for spec in build_input.system.species if isinstance(spec, LipidSpecies)
    }


def water_resnames(build_input: "BuildInput") -> list[str]:
    """Return residue names that correspond to water species."""
    return [spec.resname for spec in build_input.system.species if getattr(spec, "is_water", False)]


def residue_atoms(residue: "Residue", atom_indices: Iterable[int]) -> Any | None:
    """Return AtomGroup for indices present in the residue."""
    valid = [idx for idx in atom_indices if 0 <= idx < len(residue.atoms)]
    if not valid:
        return None
    return residue.atoms[valid]


def residue_mean_position(residue: "Residue", atom_indices: Iterable[int]) -> np.ndarray | None:
    """Return the mean position of atom indices in a residue."""
    atoms = residue_atoms(residue, atom_indices)
    if atoms is None or len(atoms) == 0:
        return None
    return atoms.positions.mean(axis=0)


def residue_atom_positions(residue: "Residue", atom_indices: Iterable[int]) -> np.ndarray:
    """Return stacked positions for the requested atom indices in a residue."""
    atoms = residue_atoms(residue, atom_indices)
    if atoms is None or len(atoms) == 0:
        return np.empty((0, 3))
    return atoms.positions


def headgroup_positions(
    universe: "mda.Universe",
    resname: str,
    head_atoms: Iterable[int],
) -> np.ndarray:
    """Return mean headgroup positions for each residue of a species."""
    residues = universe.select_atoms(f"resname {resname}").residues
    positions = []
    for residue in residues:
        pos = residue_mean_position(residue, head_atoms)
        if pos is not None:
            positions.append(pos)
    if not positions:
        return np.empty((0, 3))
    return np.vstack(positions)


def tail_positions(
    universe: "mda.Universe",
    resname: str,
    tail_atoms: Iterable[int],
) -> np.ndarray:
    """Return stacked tail atom positions for a species."""
    residues = universe.select_atoms(f"resname {resname}").residues
    positions = []
    for residue in residues:
        atom_positions = residue_atom_positions(residue, tail_atoms)
        if atom_positions.size:
            positions.append(atom_positions)
    if not positions:
        return np.empty((0, 3))
    return np.vstack(positions)


def trajectory_window(
    universe: "mda.Universe",
    *,
    start_ns: float | None = None,
    end_ns: float | None = None,
    last_ns: float | None = None,
    stride: int = 1,
) -> tuple[int, int, int]:
    """Return start/stop/step indices for a trajectory window.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with loaded trajectory
    start_ns : float | None
        Start analysis from this time in nanoseconds. Ignored when
        *last_ns* is set.
    end_ns : float | None
        Stop analysis at this time in nanoseconds.
    last_ns : float | None
        Analyze the last N ns of the trajectory. Takes precedence over
        *start_ns*.
    stride : int
        Frame stride (step size)

    Returns
    -------
    tuple[int, int, int]
        (start_frame, stop_frame, step) for trajectory slicing

    """
    total_frames = len(universe.trajectory)
    dt = universe.trajectory.dt
    if dt <= 0:
        dt = 1.0

    if last_ns is not None:
        last_ps = last_ns * 1000.0
        frames_to_analyze = max(1, int(last_ps / dt))
        start_frame = max(0, total_frames - frames_to_analyze)
    elif start_ns is not None:
        start_ps = start_ns * 1000.0
        start_frame = min(total_frames, int(start_ps / dt))
    else:
        start_frame = 0

    if end_ns is not None:
        end_ps = end_ns * 1000.0
        stop_frame = min(total_frames, int(end_ps / dt) + 1)
    else:
        stop_frame = total_frames

    return start_frame, stop_frame, stride


def midplane_from_positions(positions: np.ndarray) -> float | None:
    """Return the median z-position used for leaflet assignment."""
    if positions.size == 0:
        return None
    return float(np.median(positions[:, 2]))


def frame_time_ns(ts: Any) -> float:
    """Return the current frame time in nanoseconds."""
    return ts.time / 1000.0


class _RaggedAnalysisFromFunction(AnalysisFromFunction):
    """AnalysisFromFunction variant that preserves variable-length frame outputs."""

    def _conclude(self) -> None:
        # Keep per-frame Python objects as-is instead of forcing np.asarray conversion.
        return None


def run_per_frame_analysis(
    func: Callable,
    trajectory: Any,
    *args: Any,
    **run_kwargs: Any,
) -> list:
    """Run a per-frame analysis, preserving variable-length results as a list.

    Wraps AnalysisFromFunction but overrides _conclude to prevent numpy
    conversion of results, which fails when per-frame outputs have
    variable lengths (ragged arrays).

    Parameters
    ----------
    func : callable
        Per-frame analysis function passed to AnalysisFromFunction
    trajectory : MDAnalysis trajectory
        Trajectory to iterate over
    *args
        Additional positional arguments forwarded to AnalysisFromFunction
    **run_kwargs
        Keyword arguments forwarded to ``analysis.run()`` (e.g.,
        start, stop, step, backend, n_workers)

    Returns
    -------
    list
        Per-frame results as a Python list (not coerced to ndarray)

    """
    analysis = _RaggedAnalysisFromFunction(func, trajectory, *args)
    analysis.run(**run_kwargs)
    return list(analysis.results.timeseries)
