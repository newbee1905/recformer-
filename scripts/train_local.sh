#!/bin/bash

# This script provides a convenient way to launch local training using torchrun.
# It automatically detects the number of available NVIDIA GPUs and launches one
# training process per GPU. If no GPUs are found, it defaults to a single
# process, suitable for CPU training.
#
# Usage:
#   ./train_local.sh [hydra_options]
#
# Examples:
#   # Train locally on all available GPUs
#   ./train_local.sh
#
#   # Override the batch size
#   ./train_local.sh training.batch_size=32
#

set -euo pipefail

# Activate the virtual environment
if [ -d ".venv" ]; then
	echo "Activating virtual environment from ./.venv"
	source .venv/bin/activate
else
	echo "Warning: .venv directory not found. Assuming environment is already active."
fi

export PYTHONPATH="${PYTHONPATH:-}:."

if command -v nvidia-smi &> /dev/null; then
	NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader)
else
	echo "nvidia-smi not found, defaulting to 1 process."
	NUM_GPUS=1
fi

echo "Found $NUM_GPUS device(s). Starting local training with torchrun..."
echo "By default, this script runs the 'recurrent_mini' model for quick testing."
echo "You can override this and other settings, e.g., './train_local.sh model=recurrent_small'"
echo "Any additional arguments will be passed to main.py: $@"

# Default to the 'recurrent_mini' model if no other model is specified.
# The $@ is checked to see if 'model=' is already present.
if [[ "$@" != *model=* ]]; then
  set -- "model=recurrent_mini" "$@"
fi

torchrun \
	--nproc_per_node=$NUM_GPUS \
	main.py "$@"

echo "Local training complete."
