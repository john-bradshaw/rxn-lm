"""
Script to transform a reaction dataset (reactions as txt files) to json lines file.
"""


import json
from dataclasses import dataclass
from os import path as osp

import numpy as np
from transformers import HfArgumentParser

import rxn_lm.settings

@dataclass
class Params:
    input_file: str
    shuffle: bool = True
    random_seed: int = 42


def main(params: Params):
    # 1. Read in file
    with open(params.input_file, 'r') as fo:
        d = fo.readlines()
    rng = np.random.RandomState(params.random_seed)

    # 2. process
    def map_reaction_smiles_to_json_line(reaction_smiles):
        reactants, reagents, products = reaction_smiles.split('>')
        reactants_and_reagents = reactants.split('.')
        if len(reagents):
            reactants_and_reagents += reagents.split('.')
        if params.shuffle:
            rng.shuffle(reactants_and_reagents)
        reactants_and_reagents = ".".join(reactants_and_reagents)
        out = {
            "translation": {
                rxn_lm.settings.REACTANTS_KEY: reactants_and_reagents,
                rxn_lm.settings.PRODUCTS_KEY: products
            }
        }
        return json.dumps(out)

    d = map(lambda x: x.strip(), d)  # remove white space
    d = map(map_reaction_smiles_to_json_line, d)  # convert each line

    # 3. Write out!
    ip_stem, _ = osp.splitext(params.input_file)
    op_file = f"{ip_stem}.jsonl"

    if osp.exists(op_file):
        raise RuntimeError(f"{op_file} already exists, delete if want to overwrite")
    else:
        print(f"Writing out to {op_file}...")
        with open(op_file, 'w') as fo:
            fo.writelines('\n'.join(d))
    print('Done!')


if __name__ == '__main__':
    arg_parser = HfArgumentParser(Params)
    params: Params = arg_parser.parse_args_into_dataclasses()[0]
    main(params)
    print("done!")
