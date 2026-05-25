#!/bin/bash
#SBATCH --job-name=bench-pi052-attn
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=/fsx/pepijn/logs/bench_pi052_%j.out

set -euo pipefail

cd "${LEROBOT_ROOT:-$HOME/lerobot}"

export PATH="$HOME/miniconda3/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

run() {
    echo
    echo "--- $* ---"
    python examples/benchmark/bench_pi052_step.py "$@" || true
}

# Attention parity benchmark — same shapes, different attention kernel.
run --attn eager --batch-size 8
run --attn sdpa  --batch-size 8

# Headroom benchmark — does SDPA's memory cut allow a bigger micro-batch?
run --attn sdpa  --batch-size 12
run --attn sdpa  --batch-size 16
run --attn sdpa  --batch-size 24
