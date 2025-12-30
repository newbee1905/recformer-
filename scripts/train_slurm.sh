#!/bin/bash

# This script is designed to run the training job on a SLURM cluster.
# It requests resources (GPUs, CPUs, memory) and then uses srun and torchrun
# to launch the distributed training process defined in main.py.
#
# Before submitting, please review and customize the SBATCH directives
# (e.g., --gpus, --time, --mem, --mail-user) to match your cluster's
# configuration and your needs.
#
# Usage:
#   sbatch train_slurm.sh [hydra_options]
#
# Example:
#   # Submit a job with default settings
#   sbatch train_slurm.sh
#
#   # Submit a job and override the learning rate
#   sbatch train_slurm.sh training.optimizer.lr=0.0005
#

#SBATCH --job-name=GrokMaterial
#SBATCH --account=matsim_acc23
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@example.com # CHANGE THIS

set -euo pipefail

module load NVHPC/24.9-CUDA-12.6.0
module load Anaconda3/2024.02

eval "$(conda shell.bash hook)"

conda activate rust_build_env

export LIBCLANG_PATH="$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "Build Env Ready."
echo "Rustc path: $(which rustc)"
echo "Clang path: $(which clang)"
echo "Libclang path: $LIBCLANG_PATH"

unset CONDA_PREFIX

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Add project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Distributed Training Setup (for torchrun on SLURM) 
# Get the master node's hostname
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

# Use a random free port for the master process
export MASTER_PORT=$(shuf -i 29500-65535 -n 1)

# This should match the number of GPUs requested in the SBATCH directive.
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-1}

echo "Starting SLURM job $SLURM_JOB_ID on $(hostname)"
echo "Master Node: $MASTER_ADDR, Port: $MASTER_PORT"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Arguments passed to main.py: $@"
nvidia-smi

srun torchrun \
	--nproc_per_node=$GPUS_PER_NODE \
	--nnodes=$SLURM_NNODES \
	--rdzv_id=$SLURM_JOB_ID \
	--rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
	main.py "$@"

echo "SLURM job $SLURM_JOB_ID finished."