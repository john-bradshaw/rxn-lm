from os import path as osp

from rxn_lm.utils import is_env_variable_true

IS_DEBUG_MODE = is_env_variable_true("DEBUG_MODE", False)
WANDB_KEY_FILE_PATH = osp.join(osp.dirname(__file__), '../wandb_key.txt')
VOCAB_FILE_SAVE_NAMES = {
    "vocab_file": "vocab.json"
}
OPTIMIZER_CHKPT_NAME = "optimizer.pt"
SCHEDULER_CHKPT_NAME = "scheduler.pt"
TRAIN_ENGINE_CHKPT_NAME = "train-engine.pt"
BEST_CHECKPOINT_DIR = "checkpoints-best_loss"
METRICS_FN = "metrics.json"
EARLY_STOPPER_NAME = "early_stopper.pt"
REACTANTS_KEY = "reactants"
PRODUCTS_KEY = "products"
