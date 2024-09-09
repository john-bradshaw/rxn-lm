
import torch

from rxn_lm.eval_utils import eval_submethods


def test_clean_pred_func():
    smi1 = "CC(=O)Nc1ccc(O)cc1.CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
    smi2 = "CC(C)Cc1ccc(cc1)[C@@H](C)C(O)=O.CC(=O)Nc1ccc(O)cc1"
    smi3 = "CC(C)Cc1ccc(cc1)[C@@H](C)C(O)=O"
    cleaned_smi1 = eval_submethods.clean_pred_func(smi1)
    cleaned_smi2 = eval_submethods.clean_pred_func(smi2)
    cleaned_smi3 = eval_submethods.clean_pred_func(smi3)
    assert cleaned_smi1 == cleaned_smi2, "should be the same"
    assert cleaned_smi1 != cleaned_smi3, "should not be the same"


def test_pad_last():
    input = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_op = torch.tensor([[1, 2, 3, 12], [4, 5, 6, 12], [7, 8, 9, 12]])

    output = eval_submethods.pad_last(input, 4, 12)
    assert torch.equal(output, expected_op), "should pad last"
