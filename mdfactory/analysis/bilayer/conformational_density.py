# ABOUTME: Conformational density map generation for lipid species
# ABOUTME: Creates 3D density volumes showing lipid conformational ensemble after RMSD alignment
"""Conformational density map generation for lipid species."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .utils import lipid_species_by_resname, trajectory_window

if TYPE_CHECKING:
    from mdfactory.analysis.simulation import Simulation
    from mdfactory.models.species import LipidSpecies


DEFAULT_IL_RESNAMES = ["ILN", "ILP"]


def _heavy_neighbors(mol, atom_idx: int) -> list[int]:
    atom = mol.GetAtomWithIdx(atom_idx)
    return [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1]


def _headgroup_heavy_atoms(species: LipidSpecies) -> list[int]:
    """Return heavy-atom indices for the headgroup region.

    Uses RDKit connectivity to include heavy atoms reachable from head atoms
    without traversing beyond branch atoms (branch atoms are included but not expanded).
    """
    mol = species.rdkit_molecule
    head_set = set(species.head_atoms)
    branch_set = set(species.branch_atoms)
    if not head_set:
        return []

    visited: set[int] = set()
    stack = list(head_set)
    while stack:
        idx = stack.pop()
        if idx in visited:
            continue
        visited.add(idx)
        for nbr in _heavy_neighbors(mol, idx):
            if nbr in branch_set:
                visited.add(nbr)
                continue
            if nbr not in visited:
                stack.append(nbr)
    return sorted(visited)


def _tail_heavy_atoms_from_branch(species: LipidSpecies, n_tail_atoms: int = 4) -> list[int]:
    """Return heavy-atom indices from each tail, starting at branch points.

    For each heavy neighbor of a branch atom that is not in the headgroup,
    include up to n_tail_atoms heavy atoms by graph distance from the branch.
    """
    if n_tail_atoms <= 0:
        return []

    mol = species.rdkit_molecule
    headgroup_set = set(_headgroup_heavy_atoms(species))
    tail_atoms: set[int] = set()

    for branch_idx in species.branch_atoms:
        for tail_start in _heavy_neighbors(mol, branch_idx):
            if tail_start in headgroup_set:
                continue
            # BFS outward from branch into tail subgraph (exclude headgroup atoms)
            queue = [(tail_start, 1)]
            visited = {branch_idx}
            while queue:
                idx, dist = queue.pop(0)
                if idx in visited:
                    continue
                visited.add(idx)
                if dist <= n_tail_atoms:
                    tail_atoms.add(idx)
                if dist >= n_tail_atoms:
                    continue
                for nbr in _heavy_neighbors(mol, idx):
                    if nbr in headgroup_set or nbr in visited:
                        continue
                    queue.append((nbr, dist + 1))

    return sorted(tail_atoms)


def _get_fit_atoms(species: LipidSpecies, n_tail_atoms: int = 4) -> list[int]:
    """Return atom indices for RMSD fitting.

    Uses headgroup heavy atoms plus branch atoms and the first few heavy atoms
    of each tail chain (from the branch point outward).
    """
    fit_atoms = set(_headgroup_heavy_atoms(species))
    fit_atoms.update(species.branch_atoms)
    fit_atoms.update(_tail_heavy_atoms_from_branch(species, n_tail_atoms=n_tail_atoms))
    return sorted(fit_atoms)


def _kabsch_rotation_batched(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Compute optimal rotation matrices using batched Kabsch algorithm.

    Parameters
    ----------
    P : np.ndarray
        Mobile points, shape (batch, N, 3)
    Q : np.ndarray
        Target points, shape (N, 3) - same reference for all

    Returns
    -------
    np.ndarray
        Rotation matrices, shape (batch, 3, 3)

    """
    # Center mobile points (already centered on fit atoms before calling)
    P_centered = P - P.mean(axis=1, keepdims=True)

    # Reference is already centered
    Q_centered = Q - Q.mean(axis=0)

    # Batched covariance: (batch, 3, N) @ (N, 3) -> (batch, 3, 3)
    # Using einsum: sum over atom index n
    H = np.einsum("bna,nc->bac", P_centered, Q_centered)

    # Batched SVD - numpy handles batch dimension automatically
    U, _, Vt = np.linalg.svd(H)

    # Batched rotation: R = Vt.T @ U.T
    # Vt is (batch, 3, 3), we want Vt.transpose(0,2,1) @ U.transpose(0,2,1)
    R = np.einsum("bji,bkj->bik", Vt, U)

    # Handle reflection cases
    dets = np.linalg.det(R)
    reflect_mask = dets < 0
    if np.any(reflect_mask):
        # Flip last row of Vt for reflected cases
        Vt_fixed = Vt.copy()
        Vt_fixed[reflect_mask, -1, :] *= -1
        R[reflect_mask] = np.einsum("bji,bkj->bik", Vt_fixed[reflect_mask], U[reflect_mask])

    return R


def _align_positions_batched(
    positions: np.ndarray,
    fit_indices: list[int],
    reference_fit_positions: np.ndarray,
    center_indices: list[int] | None = None,
) -> np.ndarray:
    """Align multiple molecules to reference using fit atoms (batched).

    Parameters
    ----------
    positions : np.ndarray
        All atom positions, shape (batch, N_atoms, 3)
    fit_indices : list[int]
        Atom indices to use for fitting
    reference_fit_positions : np.ndarray
        Reference positions for fit atoms, centered at origin, shape (N_fit, 3)
    center_indices : list[int] | None
        Atom indices to use for centering translations. Defaults to fit_indices.

    Returns
    -------
    np.ndarray
        Aligned positions, shape (batch, N_atoms, 3)

    """
    if center_indices is None:
        center_indices = fit_indices

    # Center each molecule on chosen atoms (translation)
    center_positions = positions[:, center_indices, :]
    centers = center_positions.mean(axis=1, keepdims=True)  # (batch, 1, 3)
    positions_centered = positions - centers

    # Get rotation matrices: (batch, 3, 3)
    mobile_fit = positions_centered[:, fit_indices, :]
    R = _kabsch_rotation_batched(mobile_fit, reference_fit_positions)

    # Apply rotations: (batch, N_atoms, 3) @ (batch, 3, 3).T -> (batch, N_atoms, 3)
    # Using einsum: for each batch, multiply positions by R^T
    aligned = np.einsum("bna,bca->bnc", positions_centered, R)

    return aligned


def _accumulate_density(
    aligned_positions: np.ndarray,
    grid: np.ndarray,
    origin: np.ndarray,
    grid_spacing: float,
) -> None:
    """Accumulate atom positions onto density grid (in-place).

    Parameters
    ----------
    aligned_positions : np.ndarray
        Aligned atom positions, shape (N, 3)
    grid : np.ndarray
        3D density grid to accumulate into
    origin : np.ndarray
        Origin of grid in real space
    grid_spacing : float
        Grid spacing in Angstroms

    """
    grid_shape = np.array(grid.shape)

    # Vectorized index calculation
    indices = ((aligned_positions - origin) / grid_spacing).astype(int)

    # Filter to valid indices (within grid bounds)
    valid_mask = np.all((indices >= 0) & (indices < grid_shape), axis=1)
    valid_indices = indices[valid_mask]

    # Unbuffered accumulation handles duplicate indices correctly
    if len(valid_indices) > 0:
        np.add.at(grid, (valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]), 1)


def _write_mrc(
    grid: np.ndarray,
    output_path: Path,
    voxel_size: float,
    origin: np.ndarray,
) -> None:
    """Write density grid to MRC format.

    Parameters
    ----------
    grid : np.ndarray
        3D density grid indexed as [x, y, z]
    output_path : Path
        Output file path
    voxel_size : float
        Voxel size in Angstroms
    origin : np.ndarray
        Origin coordinates in Angstroms (x, y, z)

    """
    import mrcfile

    # Normalize to probability density (mrcfile requires float32)
    total = grid.sum()
    if total > 0:
        normalized = (grid / total).astype(np.float32)
    else:
        normalized = grid.astype(np.float32)

    # MRC format expects array indexed as [z, y, x] (sections, rows, columns)
    # Transpose from our [x, y, z] indexing
    mrc_data = normalized.transpose(2, 1, 0)

    with mrcfile.new(str(output_path), overwrite=True) as mrc:
        mrc.set_data(mrc_data)
        mrc.voxel_size = voxel_size
        # Set origin (in Angstroms)
        mrc.header.origin.x = origin[0]
        mrc.header.origin.y = origin[1]
        mrc.header.origin.z = origin[2]


def conformational_density_map(
    simulation: Simulation,
    *,
    species_filter: list[str] | None = None,
    start_ns: float | None = 0.0,
    stride: int = 1,
    grid_spacing: float = 0.5,
    max_residues: int | None = None,
    output_prefix: str = "conformational_density",
    write_pdb: bool = False,
    pdb_stride: int = 10,
    fit_n_tail_atoms: int = 4,
    **_kwargs,
) -> list[Path]:
    """Generate conformational density maps for lipid species.

    Aligns all lipids of each species to a reference conformation and
    accumulates atom positions on a 3D grid. Output is MRC format files
    suitable for visualization in VMD, ChimeraX, or matplotlib.

    Parameters
    ----------
    simulation : Simulation
        Simulation instance with universe and build input.
    species_filter : list[str] | None
        Residue names to analyze. Default: ["ILN", "ILP"]
    start_ns : float | None
        Start time for analysis in ns.
    stride : int
        Frame stride.
    grid_spacing : float
        Grid spacing in Angstroms.
    max_residues : int | None
        Maximum number of residues to process per species (for testing).
        If None, process all residues.
    output_prefix : str
        Output filename prefix.
    write_pdb : bool
        If True, write a multi-model PDB of aligned conformations.
    pdb_stride : int
        Write every Nth aligned lipid to reduce PDB file size.
    fit_n_tail_atoms : int
        Number of heavy atoms per tail to include in RMSD fitting (from the
        branch point outward), in addition to headgroup heavy atoms.

    Returns
    -------
    list[Path]
        Paths to generated files (MRC and optionally PDB).

    """
    from loguru import logger

    u = simulation.universe

    # Determine target species
    lipid_species = lipid_species_by_resname(simulation.build_input)
    if species_filter is None:
        target_resnames = [rn for rn in DEFAULT_IL_RESNAMES if rn in lipid_species]
    else:
        target_resnames = [rn for rn in species_filter if rn in lipid_species]

    if not target_resnames:
        logger.warning("No target species found in simulation")
        return []

    start_frame, stop_frame, step = trajectory_window(u, start_ns=start_ns, stride=stride)

    output_paths = []

    for resname in target_resnames:
        species = lipid_species[resname]
        fit_indices = _get_fit_atoms(species, n_tail_atoms=fit_n_tail_atoms)
        headgroup_heavy = _headgroup_heavy_atoms(species)
        center_indices = headgroup_heavy or fit_indices
        if not fit_indices:
            logger.warning(f"{resname}: no fit atoms available; skipping")
            continue
        if not center_indices:
            logger.warning(f"{resname}: no centering atoms available; skipping")
            continue
        logger.info(
            f"{resname}: RMSD fit using {len(fit_indices)} atoms "
            f"(head_heavy={len(headgroup_heavy)}, "
            f"branch={len(species.branch_atoms)}, tail_heavy={fit_n_tail_atoms}); "
            f"centering on {len(center_indices)} atoms"
        )

        selection = u.select_atoms(f"resname {resname}")
        if len(selection) == 0:
            logger.warning(f"No atoms found for {resname}")
            continue

        residues = selection.residues
        if max_residues is not None:
            residues = residues[:max_residues]
            logger.info(f"Limited to {len(residues)} residues (max_residues={max_residues})")

        # Build reference from first frame, first residue
        u.trajectory[start_frame]
        first_residue = residues[0]
        reference_positions = first_residue.atoms.positions.copy()

        # Center reference on headgroup atoms (or fit atoms if headgroup not defined)
        reference_center = reference_positions[center_indices].mean(axis=0)
        reference_positions -= reference_center
        reference_fit_positions = reference_positions[fit_indices].copy()

        # Determine grid extent from reference
        extent = np.max(np.abs(reference_positions)) + 5.0  # Add padding
        grid_size = int(2 * extent / grid_spacing) + 1
        origin = np.array([-extent, -extent, -extent])

        # Initialize density grid
        density_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)

        logger.info(
            f"Building density for {resname}: {len(residues)} residues, "
            f"grid {grid_size}^3, spacing {grid_spacing} A"
        )

        # Pre-fetch atom groups for all residues (avoids repeated selection)
        residue_atom_groups = [res.atoms for res in residues]

        # Setup PDB output if requested (use MDAnalysis writer)
        pdb_path = None
        pdb_writer = None
        pdb_model_count = 0
        if write_pdb:
            import MDAnalysis as mda

            pdb_path = simulation.path / f"{output_prefix}_{resname}.pdb"
            # Use first residue as template for topology
            template_atoms = residue_atom_groups[0]
            pdb_writer = mda.Writer(str(pdb_path), n_atoms=len(template_atoms), multiframe=True)

        # Accumulate over trajectory (batched per frame)
        n_frames = 0
        total_frames = len(range(start_frame, stop_frame, step))
        for ts in u.trajectory[start_frame:stop_frame:step]:
            if n_frames % 10 == 0:
                logger.info(f"Processing frame {n_frames + 1}/{total_frames}")
            _ = ts

            # Stack all residue positions: (n_residues, n_atoms, 3)
            all_positions = np.stack([ag.positions for ag in residue_atom_groups])

            # Batched alignment
            aligned_all = _align_positions_batched(
                all_positions,
                fit_indices,
                reference_fit_positions,
                center_indices=center_indices,
            )

            # Flatten and accumulate: (n_residues * n_atoms, 3)
            _accumulate_density(aligned_all.reshape(-1, 3), density_grid, origin, grid_spacing)

            # Write aligned conformations to PDB (strided)
            if pdb_writer is not None:
                for i in range(0, len(aligned_all), pdb_stride):
                    # Temporarily set positions on template atoms and write
                    template_atoms.positions = aligned_all[i]
                    pdb_writer.write(template_atoms)
                    pdb_model_count += 1

            n_frames += 1

        logger.info(f"Accumulated {n_frames} frames for {resname}")

        # Write MRC file
        output_path = simulation.path / f"{output_prefix}_{resname}.mrc"
        _write_mrc(density_grid, output_path, grid_spacing, origin)
        output_paths.append(output_path)
        logger.info(f"Wrote {output_path}")

        # Finalize PDB
        if pdb_writer is not None:
            pdb_writer.close()
            output_paths.append(pdb_path)
            logger.info(f"Wrote {pdb_path} ({pdb_model_count} models)")

    return output_paths
