"""
Script to clean a reaction dataset, i.e. perform operations such as removing atom maps etc.
"""

from dataclasses import dataclass
import functools
import warnings

from rdkit import Chem
from rdkit import RDLogger
from transformers import HfArgumentParser

from rxn_lm import chem_ops

# Turn off RDKit logging (https://github.com/rdkit/rdkit/issues/2683)
RDLogger.DisableLog('rdApp.*')


@dataclass
class Params:
    input_file: str
    output_file: str
    remove_atmmap: bool = True
    canonicalize: bool = True
    remove_on_error: bool = True


def main(params: Params):

    # 1. Work out what operation we need to apply on each SMILES molecule group.
    if (params.canonicalize):
        if (not params.remove_atmmap):
            warnings.warn("Canonicalizing but not removing atom map -- note that atom map affects canonicaliztion")
        op_ = functools.partial(chem_ops.canonicalize, remove_atm_mapping=params.remove_atmmap)
    else:
        if params.remove_atmmap:
            op_ = lambda smi: Chem.MolToSmiles(chem_ops.remove_atom_map(Chem.MolFromSmiles(smi)))
        else:
            op_ = lambda x: x

    # 2. Create function for each SMILES reaction line
    def processing_func(input_line):
        parts_of_reaction = input_line.split('>')
        try:
            out = '>'.join([op_(el) if len(el) else el for el in parts_of_reaction])
        except Exception:
            if params.remove_on_error:
                out = None
            else:
                out = input_line
        return out

    # 3. load in file
    with open(params.input_file, 'r') as fo:
        d = fo.readlines()

    # 4. process
    d = map(lambda el: el.strip(), d)  # <-- remove trailing whitespace on lines
    d = filter(lambda el: len(el), d)  # <-- remove empty lines
    d = map(processing_func, d)  # <-- run ops!
    d = filter(lambda el: el is not None, d)  # <-- remove the reactions which did not work (if applicable)

    # 5. Write out!
    print(f"Writing out to {params.output_file}...")
    with open(params.output_file, 'w') as fo:
        fo.writelines('\n'.join(d))
    print('Done!')


if __name__ == '__main__':
    arg_parser = HfArgumentParser(Params)
    params: Params = arg_parser.parse_args_into_dataclasses()[0]
    main(params)
    print("done!")
