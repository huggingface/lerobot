#!/bin/bash
#SBATCH --job-name=bench-pi052-v8-bs40-dtype
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=/fsx/pepijn/logs/bench_pi052_v8_%j.out

set -euo pipefail

cd "${LEROBOT_ROOT:-$HOME/lerobot}"

export PATH="$HOME/miniconda3/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run() {
    echo
    echo "--- $* ---"
    python examples/benchmark/bench_pi052_step.py "$@" || true
}

# Confirm BS=40 fits on a single H100 with the optimizer in residency.
run --attn sdpa --batch-size 40 --compile --compile-mode default --optimizer adamw_fused

# Dtype A/B at modest batch — fp32 needs ~2x the memory of bf16, so we
# drop to BS=4 to keep both runs comparable instead of OOMing fp32.
run --attn sdpa --batch-size 4 --optimizer adamw_fused --dtype bfloat16
run --attn sdpa --batch-size 4 --optimizer adamw_fused --dtype float32
