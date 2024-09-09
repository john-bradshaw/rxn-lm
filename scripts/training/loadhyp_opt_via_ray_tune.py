"""
Loads a hyperopt run into the python interpreter, e.g., via `ipython -i loadhyp_opt_via_ray_tune.py /path/to/experiment`

One can then do things like:
```python
res = restored_tuner.get_results()
res.num_errors
df = res.get_dataframe()
df.head()
df.iloc[df['loss'].argmin()]
bst = res.get_best_result(metric='loss', mode='min')
```
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
import ray
from ray import tune

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def get_config_as_json(df_row):
    out_dict = {}
    for key_name, value in df_row.items():
        if key_name == "loss":
            print(f"loss is {value}")
        elif key_name.split('/')[0] == "config":
            config = key_name.split('/')
            if config[1] == "wandb":
                continue
            out_dict[config[1]] = value
    return json.dumps(out_dict, cls=NpEncoder)


def get_df(restored_tuner):
    return restored_tuner.get_results().get_dataframe()

def get_args():
    parser = argparse.ArgumentParser(description="Load a Ray Tune run for analysis.")
    parser.add_argument("experiment_path")
    parser.add_argument("--address", default=None, type=str)
    args = parser.parse_args()
    print(args)
    return args


def get_checkpoint_sizes_weights(df: pd.DataFrame):
    """
    Might not be exact weights, due to some being shared, but should give a rough idea of model size.
    (can also extract this info from the logs).
    """
    import torch
    from glob import glob

    out = []

    for i, row in df.iterrows():
        log_dir = row.logdir
        chkpt_pths = glob(f"{log_dir}/checkpoints-itr*/pytorch_model.bin")
        if len(chkpt_pths) == 0:
            out.append(None)
        else:
            chkpt = torch.load(chkpt_pths[0], map_location="cpu")
            out.append(sum([v.numel() for k, v in chkpt.items()]))
    return out







def main():
    args = get_args()
    print(f"Loading HP run at {args.address}")
    ray.init(address=args.address)
    restored_tuner = tune.Tuner.restore(args.experiment_path)

    return restored_tuner


if __name__ == "__main__":
    restored_tuner = main()

