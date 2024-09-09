import collections
import functools
import json
import logging
import math
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from os import path as osp
from typing import (
    Optional,
    Callable
)

import tabulate
import torch
import wandb
from ignite.engine import Events
from rdkit import RDLogger
from torch.utils import data as torch_data
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    set_seed,
    DataCollatorForSeq2Seq,
    optimization as transformers_opt,
)

from ..settings import OPTIMIZER_CHKPT_NAME, SCHEDULER_CHKPT_NAME, TRAIN_ENGINE_CHKPT_NAME, BEST_CHECKPOINT_DIR, \
    METRICS_FN, EARLY_STOPPER_NAME

RDLogger.DisableLog('rdApp.*')

from . import train_submethods
from ..tokenizer import ReactionBartTokenizer
from ..json_utils import to_serializable
from .. import utils
from ..eval_utils import eval_submethods


@dataclass
class BartParams:
    encoder_layers: int = 12
    encoder_ffn_dim: int = 4096
    encoder_attention_heads: int = 16
    decoder_layers: int = 12
    decoder_ffn_dim: int = 4096
    decoder_attention_heads: int = 16
    encoder_layerdrop: float = 0.0
    decoder_layerdrop: float = 0.0
    activation_function: str = "gelu"
    d_model: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    init_std: float = 0.02
    classifier_dropout: float = 0.0
    scale_embedding: bool = False
    use_cache: bool = True
    num_labels: int = 3
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    is_encoder_decoder: bool = True
    num_return_sequences: int = 1  # has to be <= num beams set in training arguments


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    vocab_file: str = field(metadata={"help": "The path to the vocab file"})
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics on a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics on a jsonlines file."
        },
    )

    direction: str = field(default=None, metadata={"help": "Which way to predict the reaction, e.g., forward or "
                                                           "backward (for retrosynthesis)."})
    dataloaders_num_workers: int = field(
        default=0,
        metadata={"help": "The number of processes to use for the dataloaders."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_eval_samples_for_acc_logging: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples used"
                    "when computing the accuracy to this number."
        },
    )
    max_eval_samples_for_acc_logging_test: Optional[int] = field(
        default=None,
        metadata={
            "help": "Same as `max_eval_samples_for_acc_logging` but for the final run on the test set. "
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    def __post_init__(self):
        # accepting both json and jsonl file extensions, as
        # many jsonlines files actually have a .json extension
        valid_extensions = {"json", "jsonl"}
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in valid_extensions, "`train_file` should be a jsonlines file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in valid_extensions, "`validation_file` should be a jsonlines file."

    @classmethod  # not a property due to issues... (see https://github.com/python/cpython/issues/89519)
    def get_path_variable_names(cls):
        return {"vocab_file", "train_file", "validation_file", "test_file"}


@dataclass
class TrainingArgs:
    seed: int = field(default=42, metadata={"help": "Random Seed"})

    # tasks to perform
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_pred: bool = field(default=False, metadata={"help": "Whether to run evaluation."})

    # batch size
    train_batch_size: int = field(default=16, metadata={"help": "batch size for training data."})
    val_batch_size: int = field(default=16, metadata={"help": "batch size for validation data.."})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    # Optimizer arguments
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    num_steps: int = field(default=100, metadata={"help": "Total number of training steps to perform."})
    patience: Optional[int] = field(default=None, metadata={"help": "If set then perform early stopping with this patience."
                     "(Note done on eval loss and separate to any early stopping that might be done on the ray level)."})

    # LR scheduler args
    lr_scheduler_type: str = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    lr_scheduler_warmup_ga_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps. "
                                                           "NOTE that this is gradient accumulation steps"})

    # logging args
    log_interval_for_eval_loss: int = field(default=1000,
                                            metadata={"help": "how frequently to evaluate the validation loss"
                                                              "and save checkpoints"})
    log_interval_for_acc: int = field(default=10000,
                                            metadata={"help": "how frequently to evaluate the accuracy (expensive!)."})
    chkpts_to_keep: int = field(default=21,
                                            metadata={"help": "how many of the last checkpoints to keep."})


def train_eval_loop(
        group_name: str,
        run_name: str,
        training_args: TrainingArgs,
        data_args: DataTrainingArguments,
        bart_params: BartParams,
        logger: logging.Logger,
        starting_checkpoint_dir: Optional[str]=None,
        loss_and_save_callback: Optional[Callable]=None,
        torch_num_threads=30,
        use_tqdm=True
    ):
    """

    Note:
    * assumes wandb logger has been initialized by this point
    * assumes that have been moved to a local directory specific to run such that all
        checkpoints or other outputs can be generated locally in folder.
    * get_checkpoint_dir is a function which provides the folder to load the checkpoints from. Note that
        this is used to load the model, optimizer, scheduler, and trainer engine's state,
         but that we do not _currently_:
          * re-adjust the data: i.e., we do whatever ignite does by default and so might see multiple datapoints
            for the resuming epoch
          * maintain reproducibility: the seed is reset so will not be the same as a run from scratch.
          * previous checkpoints are not deleted so the counter used by `chkpts_to_keep` is reset (i.e., err on the side
            of keeping too many checkpoints)
          * We also do not take account of previously saved pending gradient accumulation, so the gradient averaging on
            the first step might be wrong.

    :param group_name:
    :param run_name:
    :param training_args:
    :param data_args:
    :param bart_params:
    :param logger:
    :return:
    """

    # # 1. Setup seeds, device,
    set_seed(training_args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"On device: {device}")
    if starting_checkpoint_dir is not None:
        logger.info(f"attempting to start from an existing checkpoint directory: {starting_checkpoint_dir}")
    else:
        logger.info("no starting checkpoint dir provided -- so starting from scratch!")
    if loss_and_save_callback is None:
        loss_and_save_callback = utils.noop
    torch.set_num_threads(torch_num_threads)
    torch.set_num_interop_threads(torch_num_threads)
    # ^ seems to be important when running on Ray.
    # see https://discuss.ray.io/t/ray-tune-performance-decreases-with-more-cpus-per-trial/3570/3


    # # 2. Sort out data -- first half (tokenizer and datasets)
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # reactants and products.
    # ## a. preprocessing
    tokenizer = ReactionBartTokenizer(data_args.vocab_file)
    logger.info(f"Using vocab file: {data_args.vocab_file}.")
    preprocess_function = train_submethods.create_preprocess_func(
                                       tokenizer, data_args.max_source_length,
                                       padding=False,
                                       ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
                                       direction=train_submethods.TaskDirection(data_args.direction),
                                       prefix=""
                                    )

    # ## b. datasets
    # at this point we will create a second validation dataset -- validation_acc. This is the same as the regular
    # validation but may be truncated to a smaller size to ensure that it can be used for accuracy calculations without
    # taking ages,
    data_files = {}
    raw_datasets = {}
    data_limits = dict(train=data_args.max_train_samples,
                       validation=data_args.max_eval_samples,
                       validation_acc=data_args.max_eval_samples_for_acc_logging,
                       test=data_args.max_eval_samples_for_acc_logging_test)
    for name, file_name in [("train", data_args.train_file),
                            ("validation", data_args.validation_file),
                            ("validation_acc", data_args.validation_file),
                            ("test", data_args.test_file)]:
        if file_name is not None:
            data_files[name] = file_name
            raw_datasets[name] = train_submethods.ReactionData.from_jsonl_file(file_name)
            raw_datasets[name].transform = preprocess_function

            # if user has limited amount of data then truncate dataset size
            data_limit = data_limits[name]
            if data_limit is not None:
                logger.info(f"Limiting {name} dataset to {data_limit} items.")
                raw_datasets[name] = raw_datasets[name].select(range(data_limit))


    # # 3. Model
    config = BartConfig(
        vocab_size=len(tokenizer), **asdict(bart_params),
    )
    # check some of the model params against others to ensure consistent
    assert config.bos_token_id == tokenizer.bos_token_id
    assert config.eos_token_id == tokenizer.eos_token_id
    assert config.pad_token_id == tokenizer.pad_token_id
    assert config.max_position_embeddings >= max(data_args.max_target_length, data_args.max_source_length)
    model = BartForConditionalGeneration(config)
    if starting_checkpoint_dir is not None:
        model = model.from_pretrained(starting_checkpoint_dir)
        logger.info(f"loading model from pretrained at {starting_checkpoint_dir}")
    assert model.get_input_embeddings().weight.shape[0] == len(tokenizer)
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        # ^ this is required by the collate function
    model.to(device=device)
    num_params_ = sum(param.numel() for param in model.parameters())
    trainable_params_ = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f"Number model parameters: {num_params_}")
    logger.info(f"Number trainable model parameters: {trainable_params_}")
    wandb.config.num_model_params = num_params_
    wandb.config.num_trainable_model_params = trainable_params_


    # # 4. Datasets second half -- collate function and dataloaders
    #  (this happens after the model as the collate function needs to know about model to add
    # the correct initial tokens...)
    # ## a. Collate function
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    # ## b. Dataloaders
    dataloaders = {}
    batch_sizes = dict(train=training_args.train_batch_size,
                       validation=training_args.val_batch_size,
                       validation_acc=training_args.val_batch_size,
                       test=training_args.val_batch_size)
    for name, dataset in raw_datasets.items():
        logger.info(f"Number workers for dataloaders: {data_args.dataloaders_num_workers}")
        dloader = torch_data.DataLoader(
            dataset,
            batch_size=batch_sizes[name],
            collate_fn=data_collator,
            num_workers=data_args.dataloaders_num_workers
        )
        dataloaders[name] = dloader

    # # 5. Optimizer, scheduler
    optimizer = train_submethods.get_optimizer(model,
                                          training_args.learning_rate,
                                          training_args.weight_decay)
    lr_scheduler = transformers_opt.get_scheduler(training_args.lr_scheduler_type, optimizer,
                                                  training_args.lr_scheduler_warmup_ga_steps, training_args.num_steps // training_args.gradient_accumulation_steps)
    if starting_checkpoint_dir is not None:
        optimizer_state_dict_pth = osp.join(starting_checkpoint_dir, OPTIMIZER_CHKPT_NAME)
        optimizer_state_dict = torch.load(optimizer_state_dict_pth)
        optimizer.load_state_dict(optimizer_state_dict)
        logger.info(f"loaded the optimizer state dict from: {optimizer_state_dict_pth}")
        logger.debug(optimizer.state_dict())

        scheduler_state_dict_pth = osp.join(starting_checkpoint_dir, SCHEDULER_CHKPT_NAME)
        lr_scheduler_dict = torch.load(scheduler_state_dict_pth)
        lr_scheduler.load_state_dict(lr_scheduler_dict)
        logger.info(f"loaded the scheduler state dict from: {scheduler_state_dict_pth}")


    # # 6. loss and prepare batch function for Ignite
    def loss_fn(outputs, _):
        # ^ we will piggyback off ignite's `create_supervised_trainer` function -- which expects the
        # loss_fun
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return loss

    def prepare_batch(inputs, device, non_blocking):
        inputs = train_submethods.prepare_input(inputs, device)
        return inputs, None

    # # 7. Ignite trainer
    trainer = train_submethods.create_supervised_trainer(model, optimizer, loss_fn, device,
                                        prepare_batch=prepare_batch,
                                        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                                        max_grad_norm=training_args.max_grad_norm)
    trainer.logger.addHandler(logging.FileHandler("trainer_log.log"))
    if starting_checkpoint_dir is not None:
        engine_state_dict_pth = osp.join(starting_checkpoint_dir, TRAIN_ENGINE_CHKPT_NAME)
        engine_state_dict = torch.load(engine_state_dict_pth)
        trainer.load_state_dict(engine_state_dict)
        logger.info(f"loaded trainer engine state from {engine_state_dict_pth}")
    logger.info(f"trainer engine state is: {trainer.state_dict()}")

    # # 8. Create metrics function
    compute_metrics_func = eval_submethods.create_metrics_compute_func(tokenizer, clean_pred_func=eval_submethods.clean_pred_func)

    # # 9. Create early stopper if appropriate
    if training_args.patience is not None and training_args.patience > 0:
        logger.debug(f"Setting up early stopping with patience {training_args.patience}")
        early_stopper = train_submethods.EarlyStopper(
            logger,
            patience=training_args.patience,
            trainer_to_terminate=trainer
        )
        if starting_checkpoint_dir is not None:
            early_stopper_state_dict_path = osp.join(starting_checkpoint_dir, EARLY_STOPPER_NAME)
            early_stopper.load_state_dict(torch.load(early_stopper_state_dict_path))
            logger.info(f"loaded early stopper state from {early_stopper_state_dict_path}")
        logger.info(f"early stopper state is: {early_stopper.state_dict()}")
    else:
        early_stopper = None


    # # 9. Create Saver
    def save_model(save_dir, save_extras=True, metrics=None):
        model.save_pretrained(save_dir, push_to_hub=False)
        tokenizer.save_pretrained(save_dir, push_to_hub=False)

        if save_extras:
            torch.save(training_args, osp.join(save_dir, "training_args.pt"))
            torch.save(data_args, osp.join(save_dir, "data_args.pt"))
            torch.save(optimizer.state_dict(), osp.join(save_dir, OPTIMIZER_CHKPT_NAME))
            torch.save(lr_scheduler.state_dict(), osp.join(save_dir, SCHEDULER_CHKPT_NAME))
            torch.save(trainer.state_dict(), osp.join(save_dir, TRAIN_ENGINE_CHKPT_NAME))
            if early_stopper is not None:
                torch.save(early_stopper.state_dict(), osp.join(save_dir, EARLY_STOPPER_NAME))

        if metrics is not None:
            with open(osp.join(save_dir, METRICS_FN), 'w') as fo:
                json.dump(metrics, fo, default=to_serializable)


    # # 10. Set callbacks during training:
    # ## a. every gradient accumulation step call the learning rate scheduler:
    def _grad_accum_step(engine, event):
        if (event % training_args.gradient_accumulation_steps == 0) and \
                event > 0:
            return True
        return False

    @trainer.on(Events.ITERATION_COMPLETED(event_filter=_grad_accum_step))
    def update_scheduler(trainer):
        lr_scheduler.step()

    # ## b. every step log loss
    @trainer.on(Events.ITERATION_COMPLETED)
    def update_wandb(trainer):
        wandb.log(dict(loss=trainer.state.output, epoch=trainer.state.epoch,
                       **{f"lr(from_sched)_{i}":v for i,v in enumerate(lr_scheduler.get_last_lr())},
                        **{f"lr_{i}": el["lr"] for i,el in enumerate(optimizer.param_groups)},
                       num_iterations=trainer.state.iteration,
                       ))

    # b. finish training on number of steps rather than on number of epochs by using callback to through exception
    # when number of steps required is reached
    @trainer.on(Events.ITERATION_COMPLETED)
    def end_on_max_steps(trainer):
        if trainer.state.iteration > training_args.num_steps:
            logger.info("reached total number of steps")
            trainer.terminate()

    # c. every log interval log the loss:
    @trainer.on(Events.ITERATION_COMPLETED(every=training_args.log_interval_for_eval_loss))
    def log_training_loss(trainer):
        logger.info(f"Step {trainer.state.iteration} Epoch[{trainer.state.epoch}] Train Loss: {trainer.state.output:.4f}")

    # d. evaluation and save
    # we want to:
    # 1. do a cheap loss eval (validation loss only) + checkpoint every log_interval_for_eval_loss steps
    # 2. do an expensive (accuracy and validation loss) every log_interval_for_acc
    # 3. save final model at end.
    # first we will define some functions that will be useful for these callbacks.

    # we will define the evaluation as a separate function so that we can call it separately later.
    def full_eval(epoch, iteration, dataloader_to_use, name):
        """
        we may want to run this on a smaller dataloader than the regular evaluation (as its expensive to generate);
        therefore we can take in a specific dataloader and name to print out.
        """
        prediction_outputs, observed_num_examples = eval_submethods.eval_loop(dataloader_to_use,
                                                                         model,
                                                                         functools.partial(prepare_batch,
                                                                                           device=device,
                                                                                           non_blocking=None),
                                                                         eval_submethods.PredictionStepArgs(
                                                                             teacher_forcing=False,
                                                                             max_length=data_args.max_target_length,
                                                                             num_beams=data_args.num_beams),
                                                                         use_tqdm=use_tqdm
                                                                         )
        metrics, decoded_preds = compute_metrics_func(prediction_outputs, observed_num_examples)

        logger.info(f" Epoch {epoch} [Step {iteration}] Val {name} Loss: {metrics['loss']:.4f}")

        rows = []
        for i in range(1, metrics['num_inferred_return_sequences'] + 1):
            rows.append(
                [i, metrics[f"top-{i}_accuracy"], metrics[f"top-{i}_genlength"],
                 metrics[f"top-{i}_gensmileslength"]])
        logger.info(
            f"\n results {name} ({observed_num_examples} examples): \n" + tabulate.tabulate(rows, headers=["top-k", "accuracy", "genlen", "gensmileslen"]))
        wandb.log({f"eval_{name}_" + k: v for k, v in metrics.items()})
        return metrics, decoded_preds

    def teacher_forced_eval(trainer):
        prediction_outputs, observed_num_examples = eval_submethods.eval_loop(dataloaders['validation'],
                             model,
                             functools.partial(prepare_batch, device=device, non_blocking=None),
                             eval_submethods.PredictionStepArgs(teacher_forcing=True, max_length=data_args.max_target_length),
                             use_tqdm=use_tqdm
                             )
        loss = prediction_outputs.pseudo_per_item_losses.mean()
        logger.info(f"Step {trainer.state.iteration} Epoch[{trainer.state.epoch}] Val Loss: {loss:.4f}")
        wandb.log(dict(eval_loss=loss))
        metrics = dict(loss=loss, iteration=trainer.state.iteration)
        return metrics

    checkpoint_queue = collections.deque()
    def save_checkpoint(metrics, checkpoint_dir, delete_old_chkpts_if_too_many=True):
        metrics['trainer_state'] = trainer.state_dict()
        try:
            os.makedirs(checkpoint_dir, exist_ok=False)
        except FileExistsError:
            # checkpoint could already exists (e.g. on run restarts) so will skip if that's the case.
            logger.info(f"Skipping creating checkpoint folder {checkpoint_dir} as already exists!")
            return checkpoint_dir
        save_model(checkpoint_dir, metrics=metrics)
        logger.info(f"Saved model and optimizer etc. to {checkpoint_dir}")
        checkpoint_queue.append(checkpoint_dir)
        while delete_old_chkpts_if_too_many and (len(checkpoint_queue) > training_args.chkpts_to_keep):
            chkpt_to_delete = checkpoint_queue.popleft()
            shutil.rmtree(chkpt_to_delete)
            logger.info(f"Deleted checkpoints at {chkpt_to_delete} (to make space).")

        # if loss is in the metrics then will save an additional copy of the best checkpoint seen so far in another
        # directory, e.g. so can simulate early stopping.
        # note that we do this via checking the metrics.json dict rather than keeping a store in memory, so that
        # we are "robust" to run restarts. (assuming that one does not change the validation dataset...).
        if "loss" in metrics:
            current_loss = metrics["loss"]

            # if there is a current best check point see what loss it corresponds to
            if osp.exists(BEST_CHECKPOINT_DIR):
                # get prev best from metrics json:
                with open(osp.join(BEST_CHECKPOINT_DIR, METRICS_FN), "r") as fo:
                    prev_best_metrics = json.load(fo)
                prev_best_loss = prev_best_metrics["loss"]

                # note that better losses are lower:
                create_best = prev_best_loss > current_loss

                if create_best:
                    logger.info(f"New loss ({current_loss}) is better than previous ({prev_best_loss}) so overwriting best.")
                    shutil.rmtree(BEST_CHECKPOINT_DIR)
            else:
                logger.info(f"No current best checkpoint, so current checkpoint (with loss {current_loss} will be best.")
                create_best = True

            if create_best:
                os.makedirs(BEST_CHECKPOINT_DIR, exist_ok=False)
                save_model(BEST_CHECKPOINT_DIR, metrics=metrics)
                logger.info(f"Saved best loss model so far to {BEST_CHECKPOINT_DIR}")

        return checkpoint_dir

    @trainer.on(Events.STARTED | Events.ITERATION_COMPLETED(every=training_args.log_interval_for_eval_loss))
    def compute_teacher_forced_loss(trainer):
        checkpoint_dir = f"checkpoints-itr{trainer.state.iteration}"
        logger.info("starting teacher forced eval...")
        metrics = teacher_forced_eval(trainer)
        logger.info("finished teacher forced eval...")
        save_checkpoint(metrics, checkpoint_dir)
        loss_and_save_callback(
            checkpoint_dir,
            metrics
        )
        if early_stopper is not None:
            early_stopper(score=-metrics["loss"], iteration=metrics["iteration"])
        # torch.cuda.empty_cache()

    @trainer.on(Events.STARTED | Events.ITERATION_COMPLETED(every=training_args.log_interval_for_acc))
    def complete_full_eval(trainer):
        logger.info("starting full eval...")
        full_eval(trainer.state.epoch, trainer.state.iteration,
                  dataloaders["validation_acc"],
                  f"acc(sz{len(dataloaders['validation_acc']) *dataloaders['validation_acc'].batch_size})")
        logger.info("finished full eval.")
        torch.cuda.empty_cache()


    @trainer.on(Events.COMPLETED)
    def save_end(trainer):
        logger.info("saving checkpoints end.")
        checkpoint_dir = "checkpoints-final"
        save_checkpoint({}, checkpoint_dir, delete_old_chkpts_if_too_many=False)

    # # 11. Do training!
    if training_args.do_train:
        num_epochs = math.ceil(training_args.num_steps / (len(dataloaders['train']) / training_args.train_batch_size))
        # ^ steps / steps_per_epoch (with ceil to ensure get more than required)
        logger.debug(f"Max epoch num set to {num_epochs}.")
        trainer.run(dataloaders['train'], max_epochs=num_epochs)

    # # 12. Do eval!
    if training_args.do_pred:
        logger.info(f"Doing end prediction.")
        metrics, decoded_preds = full_eval("do_eval", "do_eval", dataloaders["test"], "test")
        wandb.log({"eval_" + k:v for k,v in metrics.items()})
        cleaned_preds = [eval_submethods.clean_pred_func(pred) for pred in decoded_preds]
        output_prediction_file = osp.join(f"{datetime.now().strftime('%Y%m%d-%H%M')}-generated_predictions.txt")
        with open(output_prediction_file, "w", encoding="utf-8") as fo:
            fo.write("\n".join(cleaned_preds))
        logger.info(f"written predictions to {output_prediction_file}!")

    logger.info("done!")
