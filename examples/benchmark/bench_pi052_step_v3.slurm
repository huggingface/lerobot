#!/bin/bash
#SBATCH --job-name=bench-pi052-v3
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=/fsx/pepijn/logs/bench_pi052_v3_%j.out

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

# Compile sweep: does torch.compile + SDPA give a non-trivial boost on
# top of the bare SDPA path?
run --attn sdpa --batch-size 8  --compile
run --attn sdpa --batch-size 16 --compile

# FlexAttention sweep (experimental): score_mod adds the additive bias
# in-kernel; expect a long first-step compile, then SDPA-or-better steady
# state.
run --attn flex --batch-size 8
run --attn flex --batch-size 16
