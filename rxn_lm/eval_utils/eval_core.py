import collections
import functools
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from os import path as osp
from typing import (
    Optional,
    List)

import numpy as np
import tabulate
import torch
from rdkit import RDLogger
from torch.utils import data as torch_data
from transformers import (
    BartForConditionalGeneration,
    set_seed,
    DataCollatorForSeq2Seq,
    file_utils,
    tokenization_utils_base
)

RDLogger.DisableLog('rdApp.*')  # make RDKit quiet

from ..tokenizer import ReactionBartTokenizer
from ..settings import VOCAB_FILE_SAVE_NAMES
from ..train_utils import train_submethods
from . import eval_submethods
from .. import utils

LOSS_EVAL_STR = "loss_eval"
ACC_EVAL_STR = "acc_eval"


@dataclass
class EvalArguments:
    checkpoint_path: Optional[List[str]] = field(
        metadata={
            "help": "Locations of the checkpoints to load. "
                    "If a list then will average the checkpoints (this is experimental), but load config from first."
        },
    )

    loss_eval_files: Optional[List[str]] = field(
        metadata={
            "help": "Dataset paths to compute the loss on."
        },
    )

    acc_eval_files: Optional[List[str]] = field(
        metadata={
            "help": "Dataset paths to compute accuracy on."
        },
    )

    order_invariant_acc: bool = field(
        default=True,
        metadata={
            "help": "Whether to be invariant to the order of SMILES in the prediction when computing accuracy."
        },
    )

    seed: int = field(default=42, metadata={"help": "Random Seed"})

    num_beams: Optional[int] = field(
        default=5,
        metadata={
            "help": "Number of beams to use for evaluation. Will also be the number of sequences we return."
        },
    )

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    max_target_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. "
        },
    )

    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    max_eval_samples_for_acc_logging: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes, truncate the number of evaluation examples used"
                    "when computing the accuracy to this number."
        },
    )

    dataloaders_num_workers: int = field(
        default=0,
        metadata={"help": "The number of processes to use for the dataloaders."},
    )

    direction: str = field(default="forward", metadata={"help": "Which way to predict the reaction, e.g., forward or "
                                                           "backward (for retrosynthesis). Should match whatever"
                                                            "direction the model has been trained on."})

    batch_size: int = field(default=16, metadata={"help": "batch size to use."})

    @classmethod  # not a property due to issues... (see https://github.com/python/cpython/issues/89519)
    def get_path_variable_names(cls):
        return {"checkpoint_path", "loss_eval_files", "acc_eval_files"}


def average_model_weights(checkpoint_directory_paths, device):
    """
    Averages the model weights at `checkpoint_paths` as well as doing some loose consistency checks on the vocab and
    tokenizer configs defined at this location.

    Currently experimental.

    (should only be called if at least two checkpoint_directory_paths.)
    """
    #todo: Probably OOMs on large/very many checkpoints. rewrite to be compute the average in an online manner.

    # define the kinds of files we will read in each directory
    files_to_read = {
        "tokenizer_config": tokenization_utils_base.TOKENIZER_CONFIG_FILE,
        "vocab": VOCAB_FILE_SAVE_NAMES["vocab_file"],
        "checkpoint": file_utils.WEIGHTS_NAME
    }

    # store each of the kinds of file read in from each of the directories.
    read_in = collections.defaultdict(list)

    for fpath in checkpoint_directory_paths:
        for name_, file_name in files_to_read.items():
            full_path = osp.join(fpath, file_name)
            extension_ = osp.splitext(full_path)[1]
            if extension_ == ".json":
                with open(full_path, "r") as fo:
                    read_in[name_].append(json.load(fo))
            elif extension_ == ".bin":
                state_dict = torch.load(full_path, map_location=device)
                read_in[name_].append(state_dict)
            else:
                raise NotImplementedError(f"cannot read files of type {extension_}")

    same_vocab = all(read_in["vocab"][0] == el for el in read_in["vocab"][1:])
    same_tokenizer_config = all(read_in["tokenizer_config"][0] == el for el in read_in["tokenizer_config"][1:])
    if not (same_vocab and same_tokenizer_config):
        raise RuntimeError(f"Vocab and/or tokenizer config is inconsistent among checkpoints."
                           f"Hence would be invalid to average the weights."
                           f" Same vocab: {same_vocab}, Same tokenizer config: {same_tokenizer_config}.")

    final_ckpts = {}
    for k in read_in["checkpoint"][0]:
        averaged = torch.mean(torch.stack([el[k] for el in read_in["checkpoint"]], dim=0), dim=0)
        final_ckpts[k] = averaged
    return final_ckpts


@torch.no_grad()
def eval_loop(
        eval_args: EvalArguments,
        logger: logging.Logger,
        save_preds: bool = False,
        save_losses: bool = False,
        save_encoder_embeddings_type: Optional[str] = None,
        save_input_ids: bool = False,
        torch_num_threads=30,
        use_tqdm=True,
        device_str=None
    ):
    """
    Runs evaluation loop.

    While the train_eval_loop can do eval, this function offers extended functionality, such as doing several (different)
    datasets in sequence as well as (experimental) averaging of checkpoint weights.

    Note:
    * assumes that have been moved to local directory such that the prediction results are saved in a reasonable place.

    Details on non-obvious arguments:
    :param save_preds: whether to save the predicted SMILES (in canonical form) to output files.
    :returns results: dictionary of result summaries.
    """
    # # 1. Setup seeds, device, results dict
    set_seed(eval_args.seed)
    if device_str is None:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"On device: {device}")
    torch.set_num_threads(torch_num_threads)
    results = collections.defaultdict(dict)
    # ^ dict mapping to eval types (loss or acc) to another dict (mapping from dataset paths to respective results).

    # # 2. Create the model
    main_checkpoint_pth = eval_args.checkpoint_path[0]
    logger.info(f"Loading model from pretrained at {main_checkpoint_pth}")
    model = BartForConditionalGeneration.from_pretrained(main_checkpoint_pth)
    model = model.eval()
    model = model.to(device=device)
    num_params_ = sum(param.numel() for param in model.parameters())
    results["num-params"] = num_params_
    logger.info(f"number params: {num_params_}")

    # # 3. Create the tokenizer
    logger.info(f"loading tokenizer from {main_checkpoint_pth}")
    tokenizer = ReactionBartTokenizer.from_pretrained(main_checkpoint_pth)
    assert model.get_input_embeddings().weight.shape[0] == len(tokenizer), "model and tokenizer inconsistent."
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        # ^ this is required by the collate function

    # # 4. Average the model checkpoints if multiple have been given.
    # we do a basic check that if the vocab is defined in the folder then will check that's the same
    # but otherwise the onus is on the user to make sure that the checkpoints are compatible
    # note that this code is experimental -- I'm making use of semi-private methods that may change without warning.
    num_checkpoints = len(eval_args.checkpoint_path)
    if num_checkpoints > 1:
        logger.info(f"Will try to take average of {num_checkpoints} checkpoints.")
        avg_model_state = average_model_weights(eval_args.checkpoint_path, device)
        # we'll use the transformers `_load_state_dict_into_model` func to deal with complex situations.
        # note that it does not make use of `pretrained_model_name_or_path` apart from in logging messages so do not
        # pass in a real file path here.
        model, *_ = model._load_state_dict_into_model(model, avg_model_state, "<averaged>",
                                          ignore_mismatched_sizes=False)
        model.tie_weights()  # <- we're not training so should not be neccessary but taking a defensive approach
        model.eval()
    else:
        logger.info("Single checkpoint used")

    # # 5. Create the data
    # ## a. preprocess function
    preprocess_function = train_submethods.create_preprocess_func(
                                       tokenizer, eval_args.max_source_length,
                                       padding=False,
                                       ignore_pad_token_for_loss=eval_args.ignore_pad_token_for_loss,
                                       direction=train_submethods.TaskDirection(eval_args.direction),
                                       prefix=""
                                    )

    # ## b. datasets
    data_files = collections.defaultdict(list)
    raw_datasets = collections.defaultdict(list)
    if len(set(eval_args.loss_eval_files)) != len(eval_args.loss_eval_files):
        warnings.warn(f"Duplicates in loss_eval_files. (will only return last loss)")
    if len(set(eval_args.acc_eval_files)) != len(eval_args.acc_eval_files):
        warnings.warn(f"Duplicates in acc_eval_files. (will only return last acc)")
    for name, list_of_filenames, data_limit in [(LOSS_EVAL_STR, eval_args.loss_eval_files, eval_args.max_eval_samples),
                                    (ACC_EVAL_STR, eval_args.acc_eval_files, eval_args.max_eval_samples_for_acc_logging)]:

        for file_name in list_of_filenames:
            if file_name is not None:
                data_files[name].append(file_name)

                rdata_ = train_submethods.ReactionData.from_jsonl_file(file_name)
                rdata_.transform = preprocess_function

                if data_limit is not None:
                    logger.info(f"Limiting {file_name} (part of {name}) to {data_limit} items.")
                    rdata_ = rdata_.select(range(data_limit))

                raw_datasets[name].append(rdata_)

    # ## c. Collate function
    label_pad_token_id = -100 if eval_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    # ## d. Dataloaders
    dataloaders = collections.defaultdict(list)
    for name, dataset_list in raw_datasets.items():
        for dataset in dataset_list:
            dloader = torch_data.DataLoader(
                dataset,
                batch_size=eval_args.batch_size,
                collate_fn=data_collator,
                num_workers=eval_args.dataloaders_num_workers
            )
            dataloaders[name].append(dloader)

    # # 6. Batch and metrics function
    def prepare_batch(inputs, device, non_blocking):
        inputs = train_submethods.prepare_input(inputs, device)
        return inputs, None

    logger.info(f"Order invariant accuracy computation: {eval_args.order_invariant_acc}")
    clean_pred_func = functools.partial(eval_submethods.clean_pred_func, put_in_frozen_set=eval_args.order_invariant_acc)
    compute_metrics_func = eval_submethods.create_metrics_compute_func(tokenizer,
                                                                       clean_pred_func=clean_pred_func, log=logger)

    # # 7. Do the loss datasets
    logger.debug(f"starting on the {len(eval_args.loss_eval_files)} loss datasets...")
    for i, (path, dataloader) in enumerate(zip(eval_args.loss_eval_files, dataloaders[LOSS_EVAL_STR])):
        prediction_outputs, observed_num_examples = eval_submethods.eval_loop(dataloader,
                                                                              model,
                                                                              functools.partial(prepare_batch,
                                                                                                device=device,
                                                                                                non_blocking=None),
                                                                              eval_submethods.PredictionStepArgs(
                                                                                  teacher_forcing=True,
                                                                                  max_length=eval_args.max_target_length,
                                                                                  manually_compute_per_item_losses=save_losses,
                                                                                  collect_encoder_outputs_type=save_encoder_embeddings_type,
                                                                                  collect_input_ids=save_input_ids,
                                                                              ),
                                                                                  use_tqdm=use_tqdm,

                                                                              )
        loss = prediction_outputs.pseudo_per_item_losses.mean()
        logger.info(f"Teacher forced loss ({path}): {loss}")
        results[LOSS_EVAL_STR][path] = loss
        path_name = osp.splitext(osp.basename(path))[0]
        prefix = f"{datetime.now().strftime('%Y%m%d-%H%M')}-{path_name}"
        if save_losses:
            fname = f"{prefix}-losses-{i}.npy"
            logger.info(f"Losses for {path} being saved to {fname}.")
            np.save(fname, prediction_outputs.manually_computed_losses_per_item_per_token)
        if save_encoder_embeddings_type:
            fname = f"{prefix}-encoder_embeddings-{i}.npy"
            logger.info(f"Encoder last layer embeddings for {path} being saved to {fname}. (zero'd out with attentions)")
            np.save(fname, prediction_outputs.encoder_attn_zerod_outputs)
        if save_input_ids:
            fname = f"{prefix}-input_ids-{i}.npy"
            logger.info(f"Input ids for {path} being saved to {fname}. ")
            np.save(fname, prediction_outputs.input_ids)

    # # 8. Do the accuracy datasets
    logger.debug(f"starting on the {len(eval_args.acc_eval_files)} acc datasets...")
    for i, (path, dataloader) in enumerate(zip(eval_args.acc_eval_files, dataloaders[ACC_EVAL_STR])):
        prediction_outputs, observed_num_examples = eval_submethods.eval_loop(dataloader,
                                                                         model,
                                                                         functools.partial(prepare_batch,
                                                                                           device=device,
                                                                                           non_blocking=None),
                                                                         eval_submethods.PredictionStepArgs(
                                                                             teacher_forcing=False,
                                                                             max_length=eval_args.max_target_length,
                                                                             num_beams=eval_args.num_beams,
                                                                             num_return_sequences=eval_args.num_beams
                                                                         ),
                                                                         use_tqdm=use_tqdm
                                                                         )
        logger.debug(f"finished `eval_loop` ({i} of {len(eval_args.acc_eval_files)}), now computing metrics...")
        metrics, decoded_preds = compute_metrics_func(prediction_outputs, observed_num_examples)
        logger.debug("finished computing metrics.")
        rows = []
        for i in range(1, metrics['num_inferred_return_sequences'] + 1):
            rows.append(
                [i, metrics[f"top-{i}_accuracy"], metrics[f"top-{i}_genlength"],
                 metrics[f"top-{i}_gensmileslength"]])
        logger.info(
            f"\n results {path} ({observed_num_examples} examples):\n" + tabulate.tabulate(rows, headers=["top-k",
                                                                                                           "accuracy",
                                                                                                           "genlen",
                                                                                                           "gensmileslen"]))
        results[ACC_EVAL_STR][path] = metrics

        if save_preds:
            cleaned_preds = [eval_submethods.clean_pred_func(pred, put_in_frozen_set=False) for pred in decoded_preds]
            path_name = osp.splitext(osp.basename(path))[0]
            fname = f"{datetime.now().strftime('%Y%m%d-%H%M')}-{path_name}-{i}.txt"
            logger.info(f"Predictions for {path} being saved to {fname}.")
            with open(fname, 'w', encoding="utf-8") as fo:
                fo.write("\n".join(cleaned_preds))

    # # 9. Save prediction results
    fname = f"{datetime.now().strftime('%Y%m%d-%H%M')}-evalresults.json"
    with open(fname, "w") as fo:
        json.dump(results, fo, cls=utils.NumpyFloatValuesEncoder)
    logger.info(f"Results written to {fname} (directory: {os.getcwd()})")

    logger.info("done!")
    return results
