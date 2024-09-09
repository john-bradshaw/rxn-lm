
import copy
import datetime
import functools
import itertools
import json
import os
import warnings
from dataclasses import dataclass, asdict
from os import path as osp
from typing import Optional

import tabulate
import wandb
import ray
from ray import (
    tune,
    air)
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.air.integrations.wandb import setup_wandb
# ^ note before this was using the mixin decorator but this had a bug that same workers would reuse sessions (
# and cause https://github.com/ray-project/ray/issues/28919) -- this would cause a stack error as would get wandb
# to error when the parameters were changed.
from ray.tune import Callback
from transformers import (
    HfArgumentParser,
)

from . import train_submethods
from . import training_core
from .. import settings
from .. import utils


def single_run(group_name, config_pth="config.json", run_name="", local_dir_name="", exist_ok=True, torch_num_threads=20):
    # # Get run name:
    if not bool(run_name):
        run_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}-{'-'.join(utils.get_name())}"

    # # Setup wandb logging
    # note that this gets set up before the working directory change! (so these logs can be kept together).
    wandb.init(group=group_name,
               name=run_name,
               mode="disabled" if settings.IS_DEBUG_MODE else None)
    wandb.config.debug_mode = settings.IS_DEBUG_MODE

    # # Create and move to run
    old_wd = os.getcwd()
    folder_path = osp.join(local_dir_name, group_name, run_name)
    os.makedirs(folder_path, exist_ok=exist_ok)
    os.chdir(folder_path)

    # # Setup regular Logging
    logger = utils.get_logger(internal_log_name=run_name)
    logger.info(f"Group name: {group_name}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Working directory has now been set to {os.getcwd()}")
    logger.debug(f"Debug mode: {settings.IS_DEBUG_MODE}")

    # # Get parameters
    parser = HfArgumentParser((training_core.TrainingArgs, training_core.DataTrainingArguments, training_core.BartParams))
    config_pth_relative = osp.join(old_wd, config_pth)
    logger.info(f"params will be loaded from {config_pth_relative}")
    params = _load_params(config_pth_relative, logger=logger)
    training_args, data_args, bart_params = parser.parse_dict(params)
    wandb.config.update(dict(data_args=asdict(data_args), training_args=asdict(training_args),
                           bart_params=asdict(bart_params)))


    # # Run!
    try:
        result = training_core.train_eval_loop(
            group_name, run_name, training_args, data_args, bart_params, logger, torch_num_threads=torch_num_threads
        )
    finally:
        # put folder back regardless!
        os.chdir(old_wd)
        wandb.finish()  # will do this automatically but will call explicitly.

    return result


@dataclass
class HPTuningParams:
    config_pth: str = "config.json"
    search_space_config_pth: str = "search_spaces/default_search_space.json"
    experiment_name: str = "hp_tuning_run"
    address: Optional[str] = None
    num_samples: Optional[int] = 10
    num_cpus: Optional[int] = 30
    num_total_cpus: Optional[int] = None
    local_dir_name: str = "./ray_results"


def resume_hp_tuning(ray_path, ray_address):
    # # Set up Ray
    print(f"Ray address: {ray_address}")
    ray.init(address=ray_address)

    # # Restore tuner
    tuner = tune.Tuner.restore(
        ray_path,
        resume_unfinished=True,
        restart_errored=True,
    )

    # # Run!
    results = tuner.fit()
    print("Best result:")
    print(results.get_best_result(metric="loss"))

    return results


def _load_params(config_path, verbose=True, logger=None):
    """Load parameters and convert to absolute paths (when relevant)"""

    # # Set up logger.
    if logger is not None:
        print_ = logger.info
    else:
        print_ = print

    # # Load params
    with open(config_path, 'r') as fo:
        params = json.load(fo)

    # # Get the folder of the config -- this will be useful when converting relative paths later.
    folder_name = osp.abspath(osp.dirname(config_path))

    # # Load base params (if specified)
    # More info: if __**load_base_from_other_param special key is defined we load the params from the other config and update this one -- this is useful
    # if we have a base config which contains the core config and then we have a series of configs which only
    # specify differences from this.
    if "__**load_base_from_other_params" in params:
        logger.info(f"__**load_base_from_other_params is in params, will load in base params from here first.")
        _base_params_path = params["__**load_base_from_other_params"]
        if not osp.isabs(_base_params_path):
            _base_params_path = osp.join(folder_name, _base_params_path)
        logger.info(f"Loading base params from {_base_params_path}")

        other_params = _load_params(_base_params_path, verbose=verbose, logger=logger)

        data_params_to_change = training_core.DataTrainingArguments.get_path_variable_names() & set(params.keys())
        # ^ don't want to change keys that were changed by inner func calls (although they should be absolute paths already)

        logger.info(f"the config at: __**load_base_from_other_params, has led to the following params: {other_params.keys()}")
        other_params.update(params)
        params = other_params
    else:
        data_params_to_change = training_core.DataTrainingArguments.get_path_variable_names()

    # # Convert relative paths to absolute paths
    pth_changes = utils.set_paths_relative_to_given_pth(params, folder_name, data_params_to_change)
    if verbose:
        print_("Set training arg paths relative to original working directory ")
        table_ = tabulate.tabulate(itertools.chain(*[itertools.zip_longest((k,), v, fillvalue="")
                                                     for k, v in pth_changes.items()]),
                                   headers=["path_name", "old_path", "new_path"])
        print_(f"path changes:\n{table_}")
    return params


def run_hp_tuning(hp_tuning_params: HPTuningParams):
    """
    Run Hyperparameter optimization using Ray Tune.

    Note that we currently assume that this is run on a shared file system (or locally) so no sync is required.
    """

    # # Load parameters and convert to absolute paths (when relevant)
    params = _load_params(hp_tuning_params.config_pth)
    if params.get("patience", None) is not None:
        warnings.warn("Early stopping set on the core runner. This may interfere with any early stopping done on the "
                      "Ray tuner level.")

    # # Define search space
    with open(hp_tuning_params.search_space_config_pth, 'r') as fo:
        search_space = json.load(fo)
    search_space = train_submethods.tune_search_space_from_json(search_space)
    search_space = {
        **search_space,

        # then wandb params:
        "wandb": {
            "project": os.environ.get("WANDB_PROJECT", "rxn-lm-train"),
            "group": hp_tuning_params.experiment_name,
            "mode": "disabled" if settings.IS_DEBUG_MODE else None,
            "api_key_file": settings.WANDB_KEY_FILE_PATH
        }
    }

    # # Set up parts required by tune
    max_iters = params.get("num_steps", training_core.TrainingArgs.num_steps)
    scheduler = ASHAScheduler(metric="loss",
                              mode="min",
                              time_attr="iteration",
                              grace_period=min(20000, int(max_iters/2.)),
                              max_t=max_iters
                              )

    # todo: consider adding search algorithm

    # so decided to just use functools partial rather than `tune.with_parameters`, this is because the other parameters
    # which we're passing are small -- a small parameter dict and a string, so seems fine to pickle and replicate.
    # Also think this should be nicer on restore, where I don't need to manually respecify these parameters and rely
    # on an alpha interface. also was running into a weird bug where it seemed ray's cache got full
    # and so would fail midway through (maybe related to https://github.com/ray-project/ray/issues/30473#event-8002208730)
    # which hopefully this might avoid ...
    train_func = functools.partial(
        tune_single_run,
        base_config=params,
        group_name=hp_tuning_params.experiment_name
    )

    os.makedirs(hp_tuning_params.local_dir_name, exist_ok=True)
    run_config = air.RunConfig(
        local_dir=hp_tuning_params.local_dir_name,
        sync_config=tune.SyncConfig(
            syncer=None  # Disable syncing
        ),
        checkpoint_config=air.CheckpointConfig(num_to_keep=2),
        name=hp_tuning_params.experiment_name,
        log_to_file=True
    )

    tune_config = tune.TuneConfig(
            num_samples=hp_tuning_params.num_samples,
            scheduler=scheduler,
            reuse_actors=False
            # ^ seems safer to start each trial in a new actor to ensure resources cleaned up. There seems to be an
            # issue sometimes where wandb throws an error which derails the next run as it tries to write something
            # after the job has been killed. Other possible fixes could try is to see if can get wandb to block before
            # doing session report (and getting process potentially killed for results that are not good enough)  or
            # even if doing a sleep before the report works... (report is not called that frequently so maybe this is
            # would be okay).
        )

    # # Set up Ray
    print(f"Ray address: {hp_tuning_params.address}")
    # Will be explicit about if we are initializing Ray with a total number of CPUs.
    if hp_tuning_params.num_total_cpus is None:
        print("Initializing Ray with default number of cpus.")
        ray.init(address=hp_tuning_params.address)
    else:
        print(f"Initializing Ray. Num cpus: {hp_tuning_params.num_cpus}, Num total cpus: {hp_tuning_params.num_total_cpus}")
        ray.init(address=hp_tuning_params.address, num_cpus=hp_tuning_params.num_total_cpus)


    # # Set up tuner
    tuner = tune.Tuner(
        tune.with_resources(train_func,
                            {"gpu": 1, "cpu": hp_tuning_params.num_cpus}
                            ),
        tune_config=tune_config,
        run_config=run_config,
        param_space=search_space,
    )

    # # Run!
    results = tuner.fit()
    print("Best result:")
    print(results.get_best_result(metric="loss", mode="min"))

    return results


def tune_single_run(config, base_config, group_name="unknown_group"):
    """

    Note that this function expects that all paths are absolute. (including those set in `config` or in the config json
    files. (because Ray might change the directory and break relative paths)

    :param config:
    :param base_config_pth_abs:
    :param group_name:
    :return:
    """
    # will print out so know run starting but after will switch to using logs
    print(f"tune single run starting: {group_name}, {session.get_trial_name()}")

    # Wandb setup
    # will use fed in group as it seems on Tuner restores, the experiment name might be null...
    setup_wandb(config, allow_val_change=False, resume="allow", group=group_name)
    print("wandb setup.")

    # # Copy the config.
    # deep copy should be unnecessary but will do it in case internals of HF modify params in place
    params = copy.deepcopy(base_config)

    # # Get the run name
    run_name = session.get_trial_name()

    # # Setup regular Logging
    logger = utils.get_logger(internal_log_name=run_name)
    logger.info(f"Group name: {group_name}")
    if wandb.run.group != group_name:
        logger.warn(f"Note that the Wandb Group Name {wandb.run.group}, does not match the passed in group name {group_name}.")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Experiment name {session.get_experiment_name()}")
    logger.info(f"Trial name {session.get_trial_name()}")
    logger.info(f"Trial id {session.get_trial_id()}")
    logger.info(f"Working directory has been set to {os.getcwd()}")
    logger.info(f"Config {json.dumps(config, indent=4)}")

    # # Load in the default config, overwrite with Tune given params
    logger.info(f"Tune setting {len(config)} values.")
    parser = HfArgumentParser((training_core.TrainingArgs, training_core.DataTrainingArguments, training_core.BartParams))
    logger.info(f"base params are: {params}.")
    # ## we will now override the parts that have been given in as `config`, but first we shall print these out.
    keys_that_will_be_overwritten = params.keys() & config.keys()
    if len(keys_that_will_be_overwritten):
        rows = []
        for key_ in keys_that_will_be_overwritten:
            rows.append((key_, params[key_], config[key_]))

        logger.info(f"Tune overriding {len(keys_that_will_be_overwritten)} values set in file config:")
        logger.info('\n' + str(tabulate.tabulate(rows, headers=["key", "old-value", "new-value"])))
    params.update(config)
    if not utils.check_paths_at_keys_absolute(params, training_core.DataTrainingArguments.get_path_variable_names()):
        raise RuntimeError("Paths not defined as absolute (which is required when running as Ray Tune due to possible"
                           "directory changes.")
    # ## having updated we can now parse these values
    training_args, data_args, bart_params = parser.parse_dict(params)
    wandb.config.update(dict(data_args=asdict(data_args), training_args=asdict(training_args),
                           bart_params=asdict(bart_params)))

    # # Write function to report loss and checkpoint back to tune -- this will happen whenever we do teacher forced loss.
    def loss_and_save_callback(checkpoint_folder, metrics):
        checkpoint = Checkpoint.from_directory(checkpoint_folder)
        session.report(metrics, checkpoint=checkpoint)

    # # Finally do run!
    # we'll define the run command here as we might want to run it in different ways.
    num_cpus = int(session.get_trial_resources().head_cpus) or 20
    logger.info(f"Num cpus inferred on ray runner: {num_cpus}.")
    logger.info(f"CPU count: {os.cpu_count()}.")

    def run_command(starting_dir):
        try:
            training_core.train_eval_loop(
                group_name, run_name, training_args, data_args, bart_params, logger, starting_dir, loss_and_save_callback,
                num_cpus, False
            )
        finally:
            wandb.finish()
            print("wandb finished!")
            logger.info(f"Run command complete.")

    # so if we want to resume from a checkpoint we will run the whole training loop in the starting context.
    # as Ray might only create the checkpoint directory as a temporary thing.
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint is not None:
        with loaded_checkpoint.as_directory() as checkpoint_dir:
            result = run_command(checkpoint_dir)  # <- Ray wants us to start from a checkpoint directory
    else:
        result = run_command(None)  # <- Ray wants us to start from scratch!
    return result
