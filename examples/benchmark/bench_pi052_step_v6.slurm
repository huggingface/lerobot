#!/bin/bash
#SBATCH --job-name=bench-pi052-v6-bs32
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=/fsx/pepijn/logs/bench_pi052_v6_%j.out

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

# BS=32 with the production settings (SDPA + compile=default).
run --attn sdpa --batch-size 32 --compile --compile-mode default
