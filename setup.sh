#!/usr/bin/env bash

# Script to setup environment variables.
# Run via `source setup.sh` or `. setup.sh` so that the environment variables are set in the current shell session.
# This script should be run from the root directory of the project so that the PYTHONPATH is set correctly.

# Get the absolute path of the current script
# https://stackoverflow.com/a/246128
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
current_path=$(pwd)
if [ "$SCRIPT_DIR" != "$current_path" ]; then
    echo "WARNING: it looks like you are not running this script from the root directory of the project.
     If so, Pythonpath will not be set correctly."
fi

echo "Putting HuggingFace in offline mode..."
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Setting up W&B"
export WANDB_PROJECT=rxn-lm
#export WANDB_LOG_MODEL=true
#export WANDB_WATCH=all
#export WANDB_MODE=offline  #

echo "Adding folder to Pythonpath..."
export PYTHONPATH=${PYTHONPATH}:$(pwd)


