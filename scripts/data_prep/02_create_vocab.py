"""
Script to create a vocabulary from a reaction dataset. The files to read should be txt files of reactions line by line.
e.g.,
```
C1CCOC1.CC(C)C[Mg+].CON(C)C(=O)c1ccc(O)nc1.[Cl-]>>CC(C)CC(=O)c1ccc(O)nc1
CN.O.O=C(O)c1ccc(Cl)c([N+](=O)[O-])c1>>CNc1ccc(C(=O)O)cc1[N+](=O)[O-]
CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(N)cc3)cc21.O=CO>>CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(NC=O)cc3)cc21
```
etc.
"""

from dataclasses import dataclass
import itertools
import json
import typing

from transformers import HfArgumentParser

from rxn_lm import tokenizer


@dataclass
class Params:
    files_to_read: typing.List[str]
    vocab_fname: str = "vocab.json"


def main(params: Params):
    def create_line_generator(fname):
        def file_generator():
            with open(fname, 'r') as fo:
                for line in fo.readlines():
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    else:
                        yield line

        return file_generator()

    lines_iterator = itertools.chain(*(create_line_generator(fname) for fname in params.files_to_read))
    vocab = tokenizer.ReactionBartTokenizer.create_a_vocab(lines_iterator)

    with open(params.vocab_fname, 'w') as fo:
        json.dump(vocab, fo)
    print(f"Written out vocab to {params.vocab_fname}!")


if __name__ == '__main__':
    arg_parser = HfArgumentParser(Params)
    params: Params = arg_parser.parse_args_into_dataclasses()[0]
    main(params)
    print("done!")
