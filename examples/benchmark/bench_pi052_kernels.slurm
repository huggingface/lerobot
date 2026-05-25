#!/bin/bash
#SBATCH --job-name=bench-pi052-kernels
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=/fsx/pepijn/logs/bench_pi052_kernels_%j.out

# HF kernels exploration via Liger's apply_liger_kernel_to_paligemma.
# Baseline (SDPA, no kernels) vs. per-subkernel ablations vs. all-on.
# Same harness as bench_pi052_step.py — only the --kernels flag varies
# across runs so any delta is attributable to the patched op(s).
#
# Subkernels exercised: rope, rms_norm, geglu, layer_norm.
# Skipped: cross_entropy / fused_linear_cross_entropy — pi052 calls
# F.cross_entropy directly and bypasses PaliGemma's forward, so those
# patches wouldn't fire without model-code changes (separate PR).

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

# Liger isn't in our standard env yet — install on the compute node so
# the slurm log captures the exact version that produced the numbers.
python -m pip install -q --upgrade 'liger-kernel'
python - <<'PY' || true
from importlib.metadata import version, PackageNotFoundError
try:
    print("liger-kernel", version("liger-kernel"))
except PackageNotFoundError:
    print("liger-kernel: not importable")
import liger_kernel.transformers as t
print("apply_liger_kernel_to_paligemma:", hasattr(t, "apply_liger_kernel_to_paligemma"))
PY

run() {
    echo
    echo "--- $* ---"
    python examples/benchmark/bench_pi052_step.py "$@" || true
}

# -- Baseline (no kernels) at the BS we actually train at. --
run --attn sdpa --batch-size 8  --kernels none
run --attn sdpa --batch-size 16 --kernels none

# -- Per-subkernel ablations at BS=16 to isolate each contributor. --
run --attn sdpa --batch-size 16 --kernels rms_norm
run --attn sdpa --batch-size 16 --kernels geglu
run --attn sdpa --batch-size 16 --kernels layer_norm
run --attn sdpa --batch-size 16 --kernels rope

# -- All-on, both BS to compare against the matched baselines above. --
run --attn sdpa --batch-size 8  --kernels all
run --attn sdpa --batch-size 16 --kernels all

# -- Headroom check: does kernels-all let BS=24 fit (baseline OOMs near here)? --
run --attn sdpa --batch-size 24 --kernels none
run --attn sdpa --batch-size 24 --kernels all
