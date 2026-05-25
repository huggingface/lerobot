#!/bin/bash
#SBATCH --job-name=bench-pi052-v5
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=/fsx/pepijn/logs/bench_pi052_v5_%j.out

set -euo pipefail

cd "${LEROBOT_ROOT:-$HOME/lerobot}"

export PATH="$HOME/miniconda3/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "=== Node: $(hostname) ==="

run() {
    echo
    echo "--- $* ---"
    python examples/benchmark/bench_pi052_step.py "$@" || true
}

# compile_mode=default (graph-only, no autotune) is the right knob with
# gradient checkpointing — max-autotune in v4 was 2x slower than no-compile.
run --attn sdpa --batch-size 8  --compile --compile-mode default
run --attn sdpa --batch-size 16 --compile --compile-mode default
run --attn sdpa --batch-size 8  --compile --compile-mode reduce-overhead
