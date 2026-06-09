#!/bin/bash
#SBATCH --job-name=stream_robocasa
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Multinode streaming training over a large HF-hosted RoboCasa dataset (never touches local disk).
# Launches examples/scaling/train_streaming_multinode.py with Accelerate. Each rank streams a disjoint
# set of shards via split_dataset_by_node (auto-resolved from the Accelerate state), so per-node
# throughput scales independently. For an even split, ensure n_shards % (nodes * gpus_per_node) == 0.
#
# Submit with:  sbatch slurm/train_streaming_robocasa.sh

set -euo pipefail

REPO_ID=${REPO_ID:-pepijn223/robocasa_pretrain_human300_v4}
GPUS_PER_NODE=8
NUM_PROCESSES=$((SLURM_NNODES * GPUS_PER_NODE))

# Rendezvous: use the first node in the allocation as the main process.
MAIN_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MAIN_PORT=${MAIN_PORT:-29500}

export HF_HOME=${HF_HOME:-$SCRATCH/hf_home}
# Avoid each rank fighting over the tokenizers' internal thread pool.
export TOKENIZERS_PARALLELISM=false

srun --kill-on-bad-exit=1 bash -c '
accelerate launch \
    --num_machines '"$SLURM_NNODES"' \
    --num_processes '"$NUM_PROCESSES"' \
    --machine_rank $SLURM_NODEID \
    --main_process_ip '"$MAIN_ADDR"' \
    --main_process_port '"$MAIN_PORT"' \
    --mixed_precision bf16 \
    --dynamo_backend no \
    examples/scaling/train_streaming_multinode.py \
        --repo_id '"$REPO_ID"' \
        --batch_size 64 \
        --num_workers 12 \
        --buffer_size 4000 \
        --steps 200000 \
        --save_freq 2000 \
        --log_freq 50
'
