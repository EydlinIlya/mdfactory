# ABOUTME: Lipid nanoparticle (LNP) system construction
# ABOUTME: Builds all-atom LNP structures from composition specifications
"""Lipid nanoparticle (LNP) system construction."""

from itertools import product

import MDAnalysis as mda
import numpy as np


def pol2cart(theta, rho, z):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    cart_coords = np.stack((x, y, z), axis=1)
    return cart_coords


def cart2pol(cart_coords):
    x, y, z = cart_coords[:, 0], cart_coords[:, 1], cart_coords[:, 2]
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return theta, rho, z


def scale_flat_to_spherical(coords, radius, zo_radius):
    (theta, rho, z) = cart2pol(coords)
    total_area = 2 * np.pi * (zo_radius**2)
    areas = np.linspace(0, total_area, coords.shape[0])
    phi_sections = np.arccos(1 - (areas / total_area))
    radial_sections = radius * (np.pi / 2) * phi_sections / phi_sections.max()
    sorted_rho_ind = np.argsort(rho)
    rho[sorted_rho_ind] = radial_sections.squeeze()
    return pol2cart(theta, rho, z)


def replicate_box_3d(box: mda.Universe, n: int) -> mda.Universe:
    """Replicate a box n times in each dimension (n×n×n total copies).

    Parameters
    ----------
    box : mda.Universe
        The unit cell box to replicate.
    n : int
        Number of replications in each dimension.

    Returns
    -------
    mda.Universe
        A new universe with the replicated box.

    """
    if n == 1:
        return box.copy()

    dx, dy, dz = box.dimensions[:3]
    n_residues = len(box.residues)
    copies = []
    count = 0

    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                box_copy = box.copy()
                box_copy.residues.resids += count * n_residues
                box_copy.atoms.translate((ix * dx, iy * dy, iz * dz))
                copies.append(box_copy.atoms)
                count += 1

    merged = mda.Merge(*copies)
    merged.dimensions = [n * dx, n * dy, n * dz, 90, 90, 90]
    return merged


def replicate_layer_square(box: mda.Universe, nx: int, ny: int) -> list[mda.AtomGroup]:
    dx, dy, *_ = box.dimensions
    universes = []
    nwater = len(box.residues)
    count = 0
    for xx, yy in product(range(nx), range(ny)):
        wbox_copy = box.copy()
        wbox_copy.residues.resids += count * nwater
        wbox_copy.atoms.translate((xx * dx, yy * dy, 0))
        universes.append(wbox_copy.atoms)
        count += 1
    return universes


def shell_from_monolayer(mono: mda.Universe, R: float, z0: float = 10.0) -> mda.Universe:
    """Create a spherical shell from a monolayer by replicating the monolayer
    in x and y directions to cover a circle of radius R, then projecting
    the atoms onto a spherical surface.

    Parameters
    ----------
    mono : mda.Universe
        The input monolayer as an MDAnalysis Universe object.
    R : float
        The radius of the spherical shell in Angstroms.

    Returns
    -------
    mda.Universe
        A new MDAnalysis Universe object containing the spherical shell.

    """
    # compute the required replications to reach 2R x 2R
    dx, dy, dz, *_ = mono.dimensions
    nx = int(2 * R // dx) + 1
    ny = int(2 * R // dy) + 1
    print(f"Replicating {nx} times in x and {ny} times in y")
    boxes = replicate_layer_square(mono, nx, ny)

    big = mda.Merge(*boxes)
    big.dimensions = [nx * dx, ny * dy, dz, 90, 90, 90]

    # trim to circle of radius R
    in_circle = big.select_atoms(f"same residue as (cyzone {R} 1000 -1000 all)", periodic=False)
    # in_circle.atoms.write("circle.pdb", bonds=None)

    # center at origin
    com = in_circle.atoms.center_of_geometry()
    mins = in_circle.atoms.positions.min(axis=0)
    maxs = in_circle.atoms.positions.max(axis=0)
    in_circle.translate([-com[0], -com[1], -mins[2]])

    shell_thickness = maxs[2] - mins[2]
    # print(f"Shell thickness is {shell_thickness} A")
    print("Using pivotal plane at z =", z0)
    in_circle.atoms.positions = scale_flat_to_spherical(in_circle.atoms.positions, R, R + z0)
    # NOTE: don't move to center of geometry, as this will shift the shell up
    # (yields wrong total radius)
    # in_circle.translate(-1.0 * in_circle.center_of_geometry())
    # in_circle.atoms.write("flat2spherical.pdb", bonds=None)

    theta, rho, Z = cart2pol(in_circle.atoms.positions)

    # transform to semisphere
    radii = R + Z
    arc_length_angle = rho / R
    rho_t = radii * np.sin(arc_length_angle)
    z_t = radii * np.cos(arc_length_angle)

    # transform to Cartesian
    coords = pol2cart(theta, rho_t, z_t)

    in_circle.atoms.positions = coords.copy()
    mins = in_circle.atoms.positions.min(axis=0)
    maxs = in_circle.atoms.positions.max(axis=0)

    lower = mda.Merge(in_circle.atoms)
    rot_point = lower.atoms.center_of_geometry()
    rot_point[2] = 0.0
    lower.atoms.rotateby(angle=180, axis=[1, 0, 0], point=rot_point)
    # lower.atoms.translate([0, 0, -20.0])
    lower.atoms.residues.resids += len(in_circle.residues)

    full = mda.Merge(in_circle.atoms, lower.atoms)
    mins = full.atoms.positions.min(axis=0)
    full.atoms.translate(-1.0 * mins)
    Rtot = (R + shell_thickness) * 1.05
    full.dimensions = [2 * Rtot, 2 * Rtot, 2 * Rtot, 90, 90, 90]
    return full
