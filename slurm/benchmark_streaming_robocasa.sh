#!/bin/bash
#SBATCH --job-name=bench_stream
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Per-node dataloading benchmark for StreamingLeRobotDataset across 1-2 nodes. Each node runs an
# independent dummy-consumer benchmark; per-node throughput should be independent (separate network).
# Results are written per (node, source, mode) under --out_dir.
#
# Submit with:  sbatch slurm/benchmark_streaming_robocasa.sh
# Override the source label for cold/warm bucket runs:  SOURCE=warmed_bucket sbatch slurm/benchmark_streaming_robocasa.sh

set -euo pipefail

REPO_ID=${REPO_ID:-pepijn223/robocasa_pretrain_human300_v4}
SOURCE=${SOURCE:-hub}
OUT_DIR=${OUT_DIR:-benchmarks/streaming/results}

export HF_HOME=${HF_HOME:-$SCRATCH/hf_home}
export TOKENIZERS_PARALLELISM=false

# One benchmark process per node (each saturates the node's DataLoader workers + network independently).
srun --kill-on-bad-exit=1 bash -c '
for MODE in single sarm; do
    python benchmarks/streaming/benchmark_streaming.py \
        --repo_id '"$REPO_ID"' \
        --source '"$SOURCE"' \
        --mode $MODE \
        --batch_size 64 \
        --num_workers 12 \
        --buffer_size 4000 \
        --num_batches 300 \
        --out_dir '"$OUT_DIR"'/node${SLURM_NODEID}
done
'
