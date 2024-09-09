"""
Script for single training run.
"""

import argparse

from rxn_lm.train_utils import training_entry_points


def main():
    parser = argparse.ArgumentParser(description="Run a single run of the HF based language model using passed in args.")
    parser.add_argument("--group_name", default="hf_transformer")
    parser.add_argument("--config_pth", default="config.json")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--local_dir_name", default="")
    parser.add_argument("--torch_num_threads", default=20, type=int)
    args = parser.parse_args()

    training_entry_points.single_run(**vars(args))


if __name__ == "__main__":
    main()
