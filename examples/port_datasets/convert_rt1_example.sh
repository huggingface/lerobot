#!/bin/bash

# Example script for converting RT-1 dataset using SLURM
# Make sure to modify the paths and parameters according to your setup

# Configuration
RAW_DIR="/path/to/datasets/fractal20220817_data/0.1.0"
REPO_ID="your_username/rt1_lerobot"
LOGS_DIR="/path/to/logs"
PARTITION="cpu"  # Your SLURM partition name

# Step 1: Convert dataset using distributed processing
echo "Starting RT-1 dataset conversion..."
python examples/port_datasets/slurm_port_shards.py \
    --raw-dir "$RAW_DIR" \
    --repo-id "$REPO_ID" \
    --dataset-type rlds \
    --logs-dir "$LOGS_DIR" \
    --job-name rt1_conversion \
    --workers 32 \
    --num-shards 32 \
    --partition "$PARTITION" \
    --cpus-per-task 4 \
    --mem-per-cpu 2G \
    --slurm 1

# Step 2: Wait for jobs to complete (you can monitor with squeue)
echo "Conversion jobs submitted. Monitor with 'squeue -u \$USER'"
echo "Once all jobs complete, run the aggregation step:"
echo ""
echo "python examples/port_datasets/slurm_aggregate_shards.py \\"
echo "    --repo-id $REPO_ID \\"
echo "    --push-to-hub"

# Uncomment the following lines if you want to automatically aggregate
# (but make sure all shards are complete first)

# echo "Waiting for jobs to complete..."
# while [ $(squeue -u $USER -h | wc -l) -gt 0 ]; do
#     echo "Jobs still running, waiting 60 seconds..."
#     sleep 60
# done

# echo "All jobs completed. Starting aggregation..."
# python examples/port_datasets/slurm_aggregate_shards.py \
#     --repo-id "$REPO_ID" \
#     --push-to-hub
