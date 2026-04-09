# ABOUTME: Tests for lipid species assignment from molecular structures, including
# ABOUTME: stereochemistry handling, SMILES parsing, and atom mapping.
"""Tests for lipid species assignment from molecular structures, including."""

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from mdfactory.models.species import LipidSpecies
from mdfactory.utils.chemistry_utilities import (
    create_lipid_assignment,
    detect_lipid_parts_from_smiles_modified,
    visualize_lipid_parts_from_smiles,
)

lipid_json = Path(__file__).parent / "testfiles" / "unique_lipids.json"
with open(lipid_json, "r") as f:
    lipid_data = json.load(f)


@pytest.mark.parametrize("lipid_hash", list(lipid_data.keys()))
def test_lipid_assignment_crossref(lipid_hash, tmp_path):
    lipid_info = lipid_data[lipid_hash]
    lipid = LipidSpecies(smiles=lipid_info["smiles"], count=1, resname=lipid_info["resname"])
    assert lipid.head_atoms == lipid_info["head_atoms"]
    assert lipid.tail_atoms == sorted(lipid_info["tail_atoms"])
    assert lipid.hash == lipid_hash

    # Test the create_lipid_assignment function
    head_index, true_tail_indices, branch_indices = detect_lipid_parts_from_smiles_modified(
        lipid.smiles
    )
    mol = Chem.MolFromSmiles(lipid.smiles)
    assignment = create_lipid_assignment(
        mol,
        head_index,
        true_tail_indices,
        branch_indices,
        output_file=tmp_path / "test.png",
    )
    assert assignment is not None


def test_weird_lipids(tmp_path):
    lipid = LipidSpecies(smiles="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCS", count=1, resname="BLA")
    assert lipid.branch_atoms == []
    assert lipid.head_atoms == [31]
    assert lipid.tail_atoms == [0]

    lipid = LipidSpecies(
        smiles="CCCCCCCCCCCCCCCCC[C@H](CCCCCCCCCCCCC)CCCCCCCCCCCCCS", count=1, resname="BLA"
    )
    print(lipid.head_atoms, lipid.tail_atoms, lipid.branch_atoms)
    assert lipid.branch_atoms == [17]
    assert lipid.head_atoms == [44]
    assert lipid.tail_atoms == [0, 30]

    with pytest.warns(UserWarning, match="chemist rule"):
        lipid = LipidSpecies(smiles="CCCCCC[C@H](CCCCCCC)CCCO", count=1, resname="BLA")
    assert lipid.head_atoms == [11]
    assert lipid.tail_atoms == [0, 17]
    assert lipid.branch_atoms == [7]

    with pytest.raises(ValueError, match="Molecule graph is not fully connected"):
        lipid = LipidSpecies(smiles="CCCCCC.CCCCCCC", count=1, resname="BLA")

    lipid = LipidSpecies(
        smiles="CCCCCCCCCCCCCCCCC[C@H](CCCCCCCCCCCCC)CCCCCCCCCCCCCOCC", count=1, resname="BLA"
    )
    # mol = lipid.rdkit_molecule
    # elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    visualize_lipid_parts_from_smiles(lipid.smiles, output_file=tmp_path / "test2.png")


def test_lipid_structure_generation():
    from mdfactory.utils.setup_utilities import align_lipid_with_z_axis

    spec = LipidSpecies(
        smiles=r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC",
        resname="POC",
        count=1,
    )
    u = spec.universe
    axes = [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, -1],
    ]
    u_z = align_lipid_with_z_axis(
        u, tail_atom_ids=spec.tail_atoms, head_atom_ids=spec.head_atoms, z_axis=axes[0]
    )
    u_z2 = align_lipid_with_z_axis(
        u_z, tail_atom_ids=spec.tail_atoms, head_atom_ids=spec.head_atoms, z_axis=axes[1]
    )
    u_z3 = align_lipid_with_z_axis(
        u_z, tail_atom_ids=spec.tail_atoms, head_atom_ids=spec.head_atoms, z_axis=axes[2]
    )
    # check that the head-tail vector is aligned with z axis
    for universe, ax in zip([u_z, u_z2, u_z3], axes):
        head_pos = universe.select_atoms(f"index {spec.head_atoms[0]}").positions.mean(axis=0)
        tail_pos = universe.select_atoms(
            f"index {' '.join(map(str, spec.tail_atoms))}"
        ).positions.mean(axis=0)
        head_tail_vec = head_pos - tail_pos
        head_tail_vec /= np.linalg.norm(head_tail_vec)
        np.testing.assert_allclose(head_tail_vec, ax, atol=1e-5)

    lipid2 = LipidSpecies(
        smiles=r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[P@](=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCCCC",
        resname="POC",
        count=1,
        tail_atoms=spec.tail_atoms[::-1],
        head_atoms=spec.head_atoms,
    )
    u2 = lipid2.universe
    np.testing.assert_allclose(u.atoms.positions, u2.atoms.positions, atol=1e-5)
