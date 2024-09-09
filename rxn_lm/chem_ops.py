
from rdkit import Chem


def canonicalize(smiles, remove_atm_mapping=True, num_times=1, **otherargs) -> str:
    """
    :param smiles: SMILES string to be canonicalized
    :param remove_atm_mapping: whether to remove the atom map number (if applicable from molecule), note that this
    affects canonicalization.
    :param otherargs: arguments to pass to Chem.MolToSmiles
    :return: canonicalized SMILES string
    """
    mol = Chem.MolFromSmiles(smiles)
    out_smi = canonicalize_from_molecule(mol, remove_atm_mapping, **otherargs)
    if num_times > 1:
        for _ in range(num_times - 1):
            mol = Chem.MolFromSmiles(out_smi)
            out_smi = canonicalize_from_molecule(mol, remove_atm_mapping, **otherargs)
    return out_smi


def canonicalize_from_molecule(mol, remove_atm_mapping=True, **otherargs) -> str:
    """
    :param mol: RDKit molecule to canonicalize.
    :param remove_atm_mapping: whether to remove the atom map number (if applicable from molecule), note that this
    affects canonicalization.
    :param otherargs: arguments to pass to Chem.MolToSmiles
    :return: canonicalized SMILES string
    """
    if remove_atm_mapping:
        mol = remove_atom_map(mol)
    smiles = Chem.MolToSmiles(mol, canonical=True, **otherargs)
    return smiles


def remove_atom_map(mol):
    mol_copy = Chem.RWMol(mol)
    for atom in mol_copy.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    return mol_copy
