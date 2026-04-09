# ABOUTME: Pydantic validators for molecular input fields
# ABOUTME: Ensures canonical SMILES and valid residue names
"""Pydantic validators for molecular input fields."""

from typing import Annotated

from pydantic import AfterValidator
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers


def ensure_canonical_smiles(smi: str) -> str:
    """Validate and canonicalize a SMILES string, rejecting unspecified stereochemistry.

    Parameters
    ----------
    smi : str
        Input SMILES string.

    Returns
    -------
    str
        Canonical isomeric SMILES.

    Raises
    ------
    ValueError
        If the SMILES is invalid or has unspecified stereochemistry.

    """
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError("Molecule creation from SMILES failed.")

    isomers = tuple(EnumerateStereoisomers(m))
    if len(isomers) > 1:
        for iso in isomers:
            print(Chem.MolToSmiles(iso))
        raise ValueError(
            f"Unspecified stereochemistry: {len(isomers)} possible isomers."
            "This will lead to random placement of hydrogens and errors."
        )

    # NOTE: this does not work for spiro compounds :)
    # sinfo = Chem.FindPotentialStereo(m)
    # for si in sinfo:
    #     if not si.specified:
    #         print(
    #             f"Unspecified stereochemistry: {si.type} at atom {si.centeredOn}, {si.specified}",
    #             smi,
    #         )
    # if any(not si.specified for si in sinfo):
    #     raise ValueError(f"Molecule has Unspecified stereochemistry: {smi}")

    return Chem.MolToSmiles(m)


def validate_residue_name(resname: str) -> str:
    """Validate length and normalize a residue name to uppercase.

    Parameters
    ----------
    resname : str
        Residue name (max 3 characters).

    Returns
    -------
    str
        Uppercased residue name.

    Raises
    ------
    ValueError
        If the residue name exceeds 3 characters.

    """
    if len(resname) > 3:
        raise ValueError("Residue name must be less or equal than 3 characters long.")
    return resname.upper()


CanonicalIsomericSmiles = Annotated[str, AfterValidator(ensure_canonical_smiles)]
ResidueName = Annotated[str, AfterValidator(validate_residue_name)]
