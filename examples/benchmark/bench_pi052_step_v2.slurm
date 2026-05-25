#!/bin/bash
#SBATCH --job-name=bench-pi052-v2
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=/fsx/pepijn/logs/bench_pi052_v2_%j.out

set -euo pipefail

cd "${LEROBOT_ROOT:-$HOME/lerobot}"

export PATH="$HOME/miniconda3/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

run() {
    echo
    echo "--- $* ---"
    python examples/benchmark/bench_pi052_step.py "$@" || true
}

# A: GC ON — see if the selective-AC change (one less recompute level)
# narrows the eager vs SDPA gap at BS=8.
run --attn eager --batch-size 8
run --attn sdpa  --batch-size 8

# B: GC OFF — isolate the raw attention-kernel cost & memory delta.
run --attn eager --batch-size 4 --no-gradient-checkpointing
run --attn sdpa  --batch-size 4 --no-gradient-checkpointing

# C: SDPA + GC headroom sweep — where does it OOM?
run --attn sdpa  --batch-size 16
run --attn sdpa  --batch-size 24
run --attn sdpa  --batch-size 32
