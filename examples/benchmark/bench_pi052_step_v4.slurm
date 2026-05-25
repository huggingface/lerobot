#!/bin/bash
#SBATCH --job-name=bench-pi052-v4
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=/fsx/pepijn/logs/bench_pi052_v4_%j.out

set -euo pipefail

cd "${LEROBOT_ROOT:-$HOME/lerobot}"

export PATH="$HOME/miniconda3/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# /fsx triton cache is shared across nodes with different glibc versions
# — kernels built on one node trip GLIBC_2.34-not-found on another. Use
# a node-local cache per job to side-step that.
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
ldd --version | head -1

run() {
    echo
    echo "--- $* ---"
    python examples/benchmark/bench_pi052_step.py "$@" || true
}

# compile path on top of SDPA + selective AC
run --attn sdpa --batch-size 8  --compile
run --attn sdpa --batch-size 16 --compile

# FlexAttention experimental
run --attn flex --batch-size 8
run --attn flex --batch-size 16
