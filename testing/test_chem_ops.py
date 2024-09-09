
from rdkit import Chem

from rxn_lm import chem_ops


def test_canonicalize():
    smi = "CC1=C(C2=C(C(C=C)=C1N3CC(NCC3)C)N(C=C(C2=O)C(O)=O)CC)N"
    canonicalized_1 = chem_ops.canonicalize(smi)
    canonicalized_2 = chem_ops.canonicalize(canonicalized_1)
    assert canonicalized_1 == canonicalized_2, "should not change if canonicalized"


def test_remove_atom_maps():
    smiles_wth_atom_maps = "[CH3:1][S:207][C:200]=[C:24]([NH2:15])"
    mol = Chem.MolFromSmiles(smiles_wth_atom_maps)
    assert any(atm.HasProp('molAtomMapNumber') for atm in mol.GetAtoms())
    mol_no_maps = chem_ops.remove_atom_map(mol)
    assert not any(atm.HasProp('molAtomMapNumber') for atm in mol_no_maps.GetAtoms())