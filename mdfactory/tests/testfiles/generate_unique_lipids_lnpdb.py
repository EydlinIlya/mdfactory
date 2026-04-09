import json
from pathlib import Path

import pandas as pd
from loguru import logger
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from tqdm import tqdm

from mdfactory.models.species import LipidSpecies


def ensure_canonical_smiles(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError("Molecule creation from SMILES failed.")

    isomers = tuple(EnumerateStereoisomers(m))
    if len(isomers) > 1:
        for iso in isomers:
            return Chem.MolToSmiles(iso)
    return Chem.MolToSmiles(m)


inp = Path("LNPDB/data/LNPDB_for_LiON/LNPDB.csv")

if not inp.exists():
    raise FileNotFoundError(
        f"Input file {inp} does not exist. "
        "Please clone https://github.com/evancollins1/LNPDB in this directory."
    )


dfs = pd.read_csv(inp)

smi_cols = [
    "IL_SMILES",
    # "IL_protonated_SMILES",
    "HL_SMILES",
    "CHL_SMILES",
]

all_smi = []
for col in smi_cols:
    smi = dfs[col].dropna().unique().tolist()[:2000]
    # add tqdm progress bar

    smi = [ensure_canonical_smiles(s) for s in tqdm(smi, desc=f"Processing {col}")]
    all_smi.extend(smi)

logger.info(f"Found {len(all_smi)} SMILES strings.")
all_smi = set(all_smi)
logger.info(f"Found {len(all_smi)} unique SMILES strings.")


errors = {}
lipids = []
for smi in all_smi:
    logger.info(f"Processing SMILES: {smi}")
    try:
        ls = LipidSpecies(smiles=smi, count=1, resname="BLA")
        lipids.append(ls)
        # u = ls.universe
        # u = align_lipid_with_z_axis(u, tail_atom_ids=ls.tail_atoms,
        #                             head_atom_ids=ls.head_atoms, z_axis=[0, 0, 1])
        # u.atoms.write(f"lipids_tmp/{ls.hash}.pdb")
        # visualize_lipid_parts_from_smiles(smi, output_file=f"lipids_tmp/{ls.hash}.png")
        logger.info("OK 💕")
    except ValueError as e:
        logger.error(f"ERROR ❌: {e}")
        errors[smi] = str(e)

logger.info(f"Finished processing {len(all_smi)} unique SMILES strings.")
if errors:
    logger.error(f"Errors found in {len(errors)} SMILES strings:")
for k, v in errors.items():
    logger.error(f"{k} --> {v}")

# write erroring SMILES to a file
with open("erroring_smiles.txt", "w") as f:
    for k, v in errors.items():
        f.write(f"{k}\n")

lipid_dict = {lipid.hash: lipid.model_dump() for lipid in lipids}

with open("unique_lipids.json", "w") as f:
    f.write(json.dumps(lipid_dict, indent=None, separators=(",", ":")))
