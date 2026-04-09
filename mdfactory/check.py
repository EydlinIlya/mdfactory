# ABOUTME: Pre-build validation checks for simulation systems
# ABOUTME: Verifies that lipid species can generate valid 3D structures
"""Pre-build validation checks for simulation systems."""

from pydantic import validate_call

from .models.composition import BilayerComposition


@validate_call
def check_bilayer_buildable(system: BilayerComposition):
    """Verify that all lipid species can generate valid 3D structures.

    Parameters
    ----------
    system : BilayerComposition
        Bilayer composition to validate

    Raises
    ------
    Exception
        If any lipid species fails to generate an RDKit molecule

    """
    lipids = system.species
    for lipid in lipids:
        # try to generate a structure
        mol = lipid.rdkit_molecule
        del mol
