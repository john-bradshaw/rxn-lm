
import itertools
import json
import os
import re
from typing import Dict, Optional, Tuple, List
import warnings

from transformers import PreTrainedTokenizer
from transformers.utils import logging as t_logger

from .settings import VOCAB_FILE_SAVE_NAMES


class RegexModel:
    """
    See https://arxiv.org/abs/1711.04810; the change we made was just to extend this to include more possible loops
    """
    TOKEN_WARNING_LEVEL = 750

    def __init__(self):
        self.smiles_tokenizer_pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|\%\([0-9]{3}\)|[0-9])"
        # ^ added the 3 digit bond connection number
        self.regex_smiles_tokenizer = re.compile(self.smiles_tokenizer_pattern)

    def __call__(self, str_in):
        tokens = [token for token in self.regex_smiles_tokenizer.findall(str_in)]

        # if we have a large number of tokens, we will warn the user.
        if (len_tokens := len(tokens)) >= self.TOKEN_WARNING_LEVEL:
            warnings.warn(f"token length of {len_tokens} for {str_in}!")

        # we shall do a quick check that we can recreate the string from the tokens.
        and_back = ''.join(tokens)
        if str_in != and_back:
            raise RuntimeError(f"{str_in} was tokenized incorrectly to {tokens}")

        return tokens


class ReactionBartTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILE_SAVE_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            vocab_file,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            **kwargs
    ):

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.regex_model = RegexModel()

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text, **kwargs):
        return self.regex_model(text)

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # heavily based off the GPT-2 tokenizer in transformers library
        if not os.path.isdir(save_directory):
            t_logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILE_SAVE_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        return (vocab_file,)

    @classmethod
    def create_a_vocab(cls, molecule_lines_iterator, special_characters=None):
        if special_characters is None:
            special_characters = ["<s>", "<pad>", "</s>", "<unk>"]
        regex_model = RegexModel()
        all_tokens = set(itertools.chain(*(regex_model(molecule_section)
                                                for line in molecule_lines_iterator
                                                    for molecule_section in line.split('>'))))
        all_tokens = special_characters + sorted(list(all_tokens))
        vocab = dict(((v, i) for i, v in enumerate(all_tokens)))
        return vocab

    #todo: do we need the prefix space that GPt-2 adds? -- dont think so.

    # Below taken from t5 tokenizer
    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            import warnings
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]
