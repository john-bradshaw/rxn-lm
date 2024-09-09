"""
Script to allow you to resume a hyperparameter optimization run that was interrupted -- note this script is not fully
tested and is currently experimental.
"""
import argparse

from rxn_lm.train_utils import training_entry_points


def get_args():
    parser = argparse.ArgumentParser(description="Resume a Ray Tune run.")
    parser.add_argument("experiment_path")
    parser.add_argument("--address", default=None, type=str)
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    print("Trying to resume hp run")
    results = training_entry_points.resume_hp_tuning(args.experiment_path, args.address)
    print("back from hp run.\n")
    df = results.get_dataframe()
    print(df)
    print("\ndone!")


if __name__ == "__main__":
    main()

