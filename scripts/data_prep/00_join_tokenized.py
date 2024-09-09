"""
Optional script to join together tokenized data to turn it into a reaction SMILES file.
(i.e., you can use this if the reaction data has already been turned into source and target tokenized files and you
want to turn it back into a reaction SMILES file).
"""

import functools
from dataclasses import dataclass

from rdkit import RDLogger
from transformers import HfArgumentParser

from rxn_lm import chem_ops

# Turn off RDKit logging (https://github.com/rdkit/rdkit/issues/2683)
RDLogger.DisableLog('rdApp.*')


@dataclass
class Params:
    src_file: str
    tgt_file: str
    output_file: str
    canonicalize: bool = True
    remove_on_error: bool = True



def main(params: Params):

    # 1. Create canonicalization func
    if (params.canonicalize):
        op_ = functools.partial(chem_ops.canonicalize, remove_atm_mapping=False)
    else:
        op_ = lambda x: x

    # 2. Create function for each pair of tokenized lines
    def processing_func(src_line, tgt_line):
        out = []
        for in_ in [src_line, tgt_line]:
            merged_str = ''.join(in_.strip().split(' '))
            try:
                out.append(op_(merged_str))
            except Exception:
                if params.remove_on_error:
                    out.append(merged_str)
                else:
                    out.append(None)

        assert '>' not in out[0], "cannot currently deal with reagents in src"  # can consider coding up logic
        # for this later if want...
        out = None if None in out else '>>'.join(out)
        return out

    # 3. load in files
    src_tgt_collection = []
    for fn in [params.src_file, params.tgt_file]:

        with open(fn, 'r') as fo:
            d = fo.readlines()
            d = map(lambda el: el.strip(), d)  # <-- remove trailing whitespaces
            d = filter(len, d)  # <-- remove empty lines
            d = list(d)
            src_tgt_collection.append(d)

    assert len(src_tgt_collection[0]) == len(src_tgt_collection[1]), "src and tgt lengths do not match!"

    # 4. Process
    out = [processing_func(*el) for el in zip(*src_tgt_collection)]
    out = [el for el in out if el is not None]   # <-- remove the reactions which did not work (if applicable)

    # 5. Write out!
    print(f"Writing out to {params.output_file}...")
    with open(params.output_file, 'w') as fo:
        fo.writelines('\n'.join(out))
    print('Done!')


if __name__ == '__main__':
    arg_parser = HfArgumentParser(Params)
    params: Params = arg_parser.parse_args_into_dataclasses()[0]
    main(params)
    print("done!")

