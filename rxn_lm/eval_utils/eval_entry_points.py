
import datetime
import os
from os import path as osp
import json
import itertools
import logging
import time
import multiprocessing
import pickle
import functools

import torch
import tabulate
from transformers import HfArgumentParser

from .. import settings
from .. import utils
from . import eval_core


def eval_single_run(config_pth="eval_config.json", run_name="", local_dir_name="",
                    exist_ok=False, save_preds=True, save_losses=False,
                    save_encoder_embeddings_type=None,
                    save_input_ids=False,
                    torch_num_threads=30,
                    first_log_handler=None,
                    device_str=None,
                    use_tqdm=True):
    """
    :param config_pth: json specifying the eval run
    :param run_name: run name, if left empty will be given a random name
    :param local_dir_name: directory put before `run_name` run will take place in (and where results will be saved)
    :param exist_ok: whether okay if the run directory already exists
    :param save_preds: whether to save the predictions
    :param save_losses: whether to save the losses
    :param save_encoder_embeddings_type: whether to save the encoder embeddings and what type
    :param save_input_ids: whether to save the input ids given to method.
    :param torch_num_threads: number of cpu cores to use for run.
    :param first_log_handler: the first log handler (None will use the default handler `sys.stdout`).
    :param device_str:  device string if want to run on a specific gpu, if none will use first gpu if available and
        cpu otherwise.
    :return: result
    """
    # # Get run name:
    if not bool(run_name):
        run_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}-{'-'.join(utils.get_name())}"

    # # Create and move to run
    old_wd = os.getcwd()
    folder_path = osp.join(local_dir_name, run_name)
    os.makedirs(folder_path, exist_ok=exist_ok)
    os.chdir(folder_path)

    # # Get logger
    logger = utils.get_logger(internal_log_name=run_name, first_log_handler=first_log_handler)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Working directory has now been set to {os.getcwd()}")
    logger.debug(f"Debug mode: {settings.IS_DEBUG_MODE}")

    # # Get parameters
    parser = HfArgumentParser(eval_core.EvalArguments)
    config_pth_relative = osp.join(old_wd, config_pth)
    with open(config_pth_relative, 'r') as fo:
        params = json.load(fo)
    pth_changes = utils.set_paths_relative_to_given_pth(params,
                                                        osp.abspath(osp.dirname(config_pth_relative)),
                                                        eval_core.EvalArguments.get_path_variable_names())
    (eval_args,) = parser.parse_dict(params)
    logger.info(f"Parameters have been loaded from: {config_pth_relative}.")
    logger.info("Set training arg paths relative to original working directory ")
    table_ = tabulate.tabulate(itertools.chain(*[itertools.zip_longest((k,), v, fillvalue="")
                                                 for k, v in pth_changes.items()]),
                    headers=["path_name", "old_path", "new_path"])
    logger.debug(f"path changes:\n{table_}")

    # Run!
    try:
        result = eval_core.eval_loop(eval_args, logger, save_preds, save_losses, save_encoder_embeddings_type, save_input_ids,
                                     torch_num_threads=torch_num_threads, use_tqdm=use_tqdm, device_str=device_str)
    finally:
        os.chdir(old_wd)
    return result


def eval_multiple_runs(num_runs,
                       config_pth,
                       local_dir_name,  # for results
                       parallel_run_name,
                       single_run_config_path,
                       exist_ok,
                       save_preds,
                       save_losses,
                       save_encoder_embeddings_type,
                       save_input_ids,
                       torch_num_threads_per_run):

    # # Get run name:
    if not bool(parallel_run_name):
        parallel_run_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}-parallel-{'-'.join(utils.get_name())}"

    # # Create results directory
    local_dir_name = osp.join(local_dir_name, parallel_run_name)
    os.makedirs(local_dir_name, exist_ok=exist_ok)

    # # Create a logger (in results directory) and provide basic info
    log_name = osp.join(local_dir_name, f"{parallel_run_name}.log")
    logger = utils.get_logger(internal_log_name=parallel_run_name, log_file_name=log_name)
    logger.info(f"Parallel run name: {parallel_run_name}")
    logger.info(f"Running {num_runs} in parallel with {torch_num_threads_per_run} threads each.")
    logger.info(f"outer local directory is specified as {local_dir_name}")

    # # Create config directory for the configs
    configs_pths = osp.join(single_run_config_path, parallel_run_name)
    os.makedirs(configs_pths, exist_ok=exist_ok)
    logger.info(f"configs path is {local_dir_name}")

    # # Read in config file
    logger.info(f"Reading in parallel config file from {config_pth}")
    with open(config_pth, 'r') as fo:
        parallel_run_settings = json.load(fo)

    # # Create chkpts to run against/the remaining settings will be the config for each run.
    chkpt_dirs = parallel_run_settings.pop('chkpt_dirs')
    chkpt_to_use = parallel_run_settings.pop('chkpt_to_use')
    logger.info(f"will run on {len(chkpt_dirs)} checkpoints")

    run_config = parallel_run_settings  # <- the remainder is the config for each run.
    pth_changes = utils.set_paths_relative_to_given_pth(run_config,
                                                        osp.abspath(osp.dirname(config_pth)),
                                                        eval_core.EvalArguments.get_path_variable_names())
    logger.info("Set paths relative to config.")
    table_ = tabulate.tabulate(itertools.chain(*[itertools.zip_longest((k,), v, fillvalue="")
                                                 for k, v in pth_changes.items()]),
                    headers=["path_name", "old_path", "new_path"])
    logger.debug(f"path changes:\n{table_}")
    logger.debug(f"run_config: {run_config}")

    # # Save single function
    single_run = functools.partial(_single_run, run_config=run_config, chkpt_to_use=chkpt_to_use,
                             configs_pths=configs_pths, parallel_run_name=parallel_run_name, logger=logger,
                             local_dir_name=local_dir_name, exist_ok=exist_ok,
                            save_preds=save_preds, save_losses=save_losses,
                             save_encoder_embeddings_type=save_encoder_embeddings_type,
                             save_input_ids=save_input_ids, torch_num_threads_per_run=torch_num_threads_per_run)

    # # Run in parallel
    queue = multiprocessing.Queue()
    for i in range(num_runs):
        queue.put(i)
    with multiprocessing.Pool(processes=num_runs, initializer=_init, initargs=(queue,)) as pool:
        all_results = pool.map(single_run, chkpt_dirs, chunksize=1)
    all_results = dict(zip(chkpt_dirs, all_results))

    # # Save results
    results_name = osp.join(local_dir_name, f"{parallel_run_name}_results.pick")
    with open(results_name, "wb") as fo:
        pickle.dump(all_results, fo)
    logger.info(f"results saved to {results_name}")


def _init(queue):
    # will be used so that each single run runs on a different GPU.
    global idx
    idx = queue.get()


def _single_run(chkpt_dir,

                # the rest are always same
                run_config, chkpt_to_use, configs_pths, parallel_run_name, logger, local_dir_name, exist_ok,
                save_preds, save_losses, save_encoder_embeddings_type, save_input_ids, torch_num_threads_per_run
                ):
    global idx

    time.sleep(2)  # <- hopefully not necessary but will wait 2 secs in case old process needs time to clear out.
    new_run_config = run_config.copy()
    new_run_config["checkpoint_path"] = [osp.abspath(osp.join(chkpt_dir, chkpt_to_use))]

    final_folder_name = osp.basename(chkpt_dir)
    run_name = f"{parallel_run_name}-{final_folder_name}"
    logger.info(f"starting run: {run_name}")

    config_loc = osp.join(configs_pths, f"{run_name}.json")
    with open(config_loc, "w") as fo:
        json.dump(new_run_config, fo)
    logger.debug(f"new config file written to {config_loc}")

    if torch.cuda.is_available():
        device_str = f"cuda:{idx}"
    else:
        device_str = "cpu"
    logger.debug(f"device string is {device_str}")

    null_handler = logging.NullHandler()
    results = eval_single_run(
        config_pth=config_loc,
        run_name=run_name,
        local_dir_name=local_dir_name,
        exist_ok=exist_ok,
        save_preds=save_preds,
        save_losses=save_losses,
        save_encoder_embeddings_type=save_encoder_embeddings_type,
        save_input_ids=save_input_ids,
        torch_num_threads=torch_num_threads_per_run,
        first_log_handler=null_handler,
        device_str=device_str,
        use_tqdm=False
    )
    logger.info(f"finished run: {run_name}")

    with torch.cuda.device(device_str):
        torch.cuda.empty_cache()
    time.sleep(0.5)

    return results
