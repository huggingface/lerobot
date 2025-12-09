#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=hopper-cpu
#SBATCH --cpus-per-task=96
#SBATCH --output=/fsx/jade_choghari/logs/launcher_%j.out
#SBATCH --error=/fsx/jade_choghari/logs/launcher_%j.err
# Activate conda environment
# Load conda
source /fsx/jade_choghari/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
set -e  # Exit on error

# Input dataset
REPO_ID="lerobot"
ROOT="/fsx/jade_choghari/vlabench-primitive/"

# Output paths
OUTPUT_DIR="/fsx/jade_choghari/vlabench-primitive-encoded"
OUTPUT_REPO_ID="vlabench-primitive-encoded"
LOGS_DIR="/fsx/jade_choghari/logs/convert_video"

# Video encoding settings
VCODEC="libsvtav1"
PIX_FMT="yuv420p"
GOP_SIZE=2
CRF=30
FAST_DECODE=0

# Parallelization settings
NUM_WORKERS=24           # Number of parallel SLURM workers
NUM_IMAGE_WORKERS=4       # Threads per worker for image saving

# SLURM settings
PARTITION="hopper-cpu"  # Change to your CPU partition name
CPUS_PER_TASK=32           # CPUs per worker
MEM_PER_CPU="4G"          # Memory per CPU
TIME_LIMIT="24:00:00"     # Time limit per job

###############################################################################
# STEP 1: Parallel Video Conversion
###############################################################################
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "STEP 1: Starting parallel video conversion"
echo "  Workers: ${NUM_WORKERS}"
echo "  Input: ${REPO_ID} (root: ${ROOT})"
echo "  Output: ${OUTPUT_DIR}"
echo "=============================================="

python /admin/home/jade_choghari/lerobot/examples/port_datasets/slurm_convert_to_video.py\
    --repo-id "${REPO_ID}" \
    --root "${ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-repo-id "${OUTPUT_REPO_ID}" \
    --vcodec "${VCODEC}" \
    --pix-fmt "${PIX_FMT}" \
    --g ${GOP_SIZE} \
    --crf ${CRF} \
    --fast-decode ${FAST_DECODE} \
    --num-image-workers ${NUM_IMAGE_WORKERS} \
    --logs-dir "${LOGS_DIR}" \
    --job-name "convert_video" \
    --slurm 1 \
    --workers ${NUM_WORKERS} \
    --partition "${PARTITION}" \
    --cpus-per-task ${CPUS_PER_TASK} \
    --mem-per-cpu "${MEM_PER_CPU}" \
    --time-limit "${TIME_LIMIT}"

echo ""
echo "✓ Parallel conversion jobs submitted!"
echo "  Monitor with: squeue -u \$USER"
echo "  Check logs in: ${LOGS_DIR}/convert_video"
echo ""
echo "Wait for all jobs to complete before running Step 2."
echo "You can check completion with: squeue -u \$USER | grep convert_video"
echo ""
echo "After all jobs complete, run Step 2 to aggregate shards:"
echo "  bash convert_to_video_parallel.sh aggregate"

###############################################################################
# STEP 2: Aggregate Shards (run this after Step 1 completes)
###############################################################################

if [ "$1" == "aggregate" ]; then
    echo ""
    echo "=============================================="
    echo "STEP 2: Aggregating video shards"
    echo "  Shards: ${NUM_WORKERS}"
    echo "  Input: ${OUTPUT_DIR}/shard_XXXX"
    echo "  Output: ${OUTPUT_DIR}_final"
    echo "=============================================="

    python slurm_aggregate_video_shards.py \
        --shards-dir "${OUTPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}_final" \
        --output-repo-id "${OUTPUT_REPO_ID}" \
        --num-shards ${NUM_WORKERS} \
        --logs-dir "${LOGS_DIR}" \
        --job-name "aggregate_video" \
        --slurm 1 \
        --partition "${PARTITION}" \
        --cpus-per-task 16 \
        --mem-per-cpu "8G" \
        --time-limit "08:00:00"

    echo ""
    echo "✓ Aggregation job submitted!"
    echo "  Monitor with: squeue -u \$USER | grep aggregate_video"
    echo "  Check logs in: ${LOGS_DIR}/aggregate_video"
    echo ""
    echo "After completion, your final dataset will be in:"
    echo "  ${OUTPUT_DIR}_final"
fi

###############################################################################
# Helpful information
###############################################################################

if [ "$1" != "aggregate" ]; then
    echo ""
    echo "=============================================="
    echo "WORKFLOW SUMMARY"
    echo "=============================================="
    echo ""
    echo "1. Step 1 is now running - it will:"
    echo "   - Split episodes across ${NUM_WORKERS} workers"
    echo "   - Each worker converts its episodes to video"
    echo "   - Creates shard datasets in ${OUTPUT_DIR}/shard_XXXX"
    echo ""
    echo "2. After Step 1 completes, run Step 2:"
    echo "   bash convert_to_video_parallel.sh aggregate"
    echo ""
    echo "3. Step 2 will merge all shards into a single dataset"
    echo ""
    echo "=============================================="
fi


