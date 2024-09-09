"""
Script for running a single evaluation using one set of weights.
"""

import argparse

from rxn_lm.eval_utils import eval_entry_points


def main():
    parser = argparse.ArgumentParser(description="Run a single run of the HF based language model using passed "
                                                 "in args.")
    parser.add_argument("--config_pth", default="eval_config.json")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--local_dir_name", default="results")
    parser.add_argument("--exist_ok", action="store_true")
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--save_losses", action="store_true")
    parser.add_argument("--save_encoder_embeddings_type", default=None, choices=[None, "all", "avg"],
                        help="whether to store the encoder embeddings for the input, and if so what kind to store "
                             "(e.g., 'avg' for averaged over sequence length). Leave blank for no embeddings.")
    parser.add_argument("--save_input_ids", action="store_true")
    parser.add_argument("--torch_num_threads", default=30)
    args = parser.parse_args()

    eval_entry_points.eval_single_run(**vars(args))


if __name__ == "__main__":
    main()
