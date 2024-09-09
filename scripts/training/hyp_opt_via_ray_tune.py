"""
Script to run hyperparameter optimization using Ray tune.
"""

import argparse

from rxn_lm.train_utils import training_entry_points


def get_args():
    parser = argparse.ArgumentParser(description="Run a single run of the HF based language model using passed in args.")
    parser.add_argument("--experiment_name", default="hf_tune_run", help="experiment name (used to set group name in wandb)")
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--config_pth", default="config.json")
    parser.add_argument("--search_space_config_pth", default="search_spaces/default_search_space.json")
    parser.add_argument("--address", default=None, type=str)
    parser.add_argument("--num_cpus", default=20, type=int)
    parser.add_argument("--num_total_cpus", default=None, type=int)
    parser.add_argument("--local_dir_name", default="./ray_results", type=str)
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    params = training_entry_points.HPTuningParams(args.config_pth, args.search_space_config_pth, args.experiment_name, args.address, args.num_samples,
                                                  num_cpus=args.num_cpus, num_total_cpus=args.num_total_cpus, local_dir_name=args.local_dir_name)
    print("Calling hp run function")
    results = training_entry_points.run_hp_tuning(params)
    print("back from hp run.\n")
    df = results.get_dataframe()
    print(df)
    print("\ndone!")


if __name__ == "__main__":
    main()

