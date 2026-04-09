# ABOUTME: Simulation box dimension time series analysis
# ABOUTME: Tracks box lengths and volume over trajectory frames
"""Box dimension analysis."""

from __future__ import annotations

import pandas as pd
from MDAnalysis.analysis.base import AnalysisFromFunction

from .utils import trajectory_window


def _box_size_frame(atomgroup) -> dict[str, float | int]:
    dims = atomgroup.dimensions
    return {
        "time_ns": atomgroup.universe.trajectory.time / 1000.0,
        "frame": atomgroup.universe.trajectory.frame,
        "x": float(dims[0]),
        "y": float(dims[1]),
        "z": float(dims[2]),
        "volume": float(dims[0] * dims[1] * dims[2]),
    }


def box_size_timeseries(
    simulation,
    *,
    start_ns: float | None = None,
    stride: int = 1,
    backend: str = "multiprocessing",
    n_workers: int = 4,
) -> pd.DataFrame:
    """Record box dimensions and volume over time.

    Parameters
    ----------
    simulation : Simulation
        Simulation instance with universe and build input.
    start_ns : float | None
        Start analysis from this time in nanoseconds.
    stride : int
        Frame stride.
    backend : str
        MDAnalysis backend for parallel execution.
    n_workers : int
        Number of workers for parallel execution.

    Returns
    -------
    pd.DataFrame
        Columns: time_ns, frame, x, y, z, volume.

    """
    u = simulation.universe
    start_frame, stop_frame, step = trajectory_window(u, start_ns=start_ns, stride=stride)

    ag = u.select_atoms("all")
    analysis = AnalysisFromFunction(_box_size_frame, u.trajectory, ag)
    analysis.run(
        start=start_frame,
        stop=stop_frame,
        step=step,
        backend=backend,
        n_workers=n_workers,
    )
    return pd.DataFrame(list(analysis.results.timeseries))
