"""
Runs several hf_evals in parallel. Does this using the multiprocessing library, this means that the results can be
merged easily into one datastructure afterwards but it also means that no further subprocesses (e.g., for dataloading)
can be used.
"""
import argparse

from rxn_lm.eval_utils import eval_entry_points


def main():
    parser = argparse.ArgumentParser(description="Run a series of hf_eval runs in parallel. Each run uses different"
                                                 "weights but tests on the same data.")
    parser.add_argument("--num_runs", type=int, default=2, help="number of runs to do in parallel. "
                                                                "(uses this many gpus)")
    parser.add_argument("--config_pth", default="eval_config_parallel.json", help="config for the parallel run.")
    parser.add_argument("--local_dir_name", default="results")
    parser.add_argument("--parallel_run_name", default="")
    parser.add_argument("--single_run_config_path", default="configs", help="path to where the config"
                                                                            "jsons for each run will be created.")
    parser.add_argument("--exist_ok", action="store_true")
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--save_losses", action="store_true")
    parser.add_argument("--save_encoder_embeddings_type",  default=None, choices=[None, "all", "avg"],
                        help="whether to store the encoder embeddings for the input, and if so what kind to store "
                             "(e.g., 'avg' for averaged over sequence length). Leave blank for no embeddings.")
    parser.add_argument("--save_input_ids", action="store_true")
    parser.add_argument("--torch_num_threads_per_run", default=30, help="number of threads to use for each run.",
                        type=int)
    args = parser.parse_args()

    eval_entry_points.eval_multiple_runs(**vars(args))


if __name__ == "__main__":
    main()
