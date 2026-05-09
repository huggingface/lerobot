#!/usr/bin/env bash
# Sweep SARM 3-stage iter checkpoints on the v2 val set ONLY.
# Usage:
#   bash scripts_local/eval_sarm_3stage_v2.sh <iter_dir> [ckpts...]
# Examples:
#   bash scripts_local/eval_sarm_3stage_v2.sh sim_3stage_sarm_iter6                        # default sweep last+10k+12k+14k
#   bash scripts_local/eval_sarm_3stage_v2.sh sim_3stage_sarm_iter6 008000 014000          # specific ckpts
set -euo pipefail

ITER_DIR="${1:-}"
shift || true
CKPTS=("${@:-008000 010000 012000 014000}")
if [ -z "$ITER_DIR" ]; then
    echo "usage: eval_sarm_3stage_v2.sh <iter_dir> [ckpts...]" >&2; exit 1
fi

OUT_ROOT="outputs/sarm_eval_3stage_v2_$(basename "$ITER_DIR")"
mkdir -p "$OUT_ROOT"

DS="domrachev03/sim_3stage_v2_val_fs"
STATS="domrachev03/sim_3stage_v2_train_fs"
TASK="Three-stage assembly"

for ck in ${CKPTS[@]}; do
    PRETRAINED="outputs/$ITER_DIR/checkpoints/$ck/pretrained_model"
    if [ ! -d "$PRETRAINED" ]; then
        echo "skip $ck: $PRETRAINED missing"; continue
    fi
    OUT_DIR="$OUT_ROOT/val_fs_$ck"
    LABEL="$(basename "$ITER_DIR")_$ck"
    echo "==[ eval $LABEL on $DS ]=="
    uv run python -m lerobot_policy_sarm.eval_sarm_sim_assemble \
        --dataset "$DS" \
        --pretrained "$PRETRAINED" \
        --stats "$STATS" \
        --task "$TASK" \
        --type sarm_ext \
        --head-mode sparse \
        --mode sync \
        --device cuda \
        --success-threshold 0.95 \
        --image-key observation.images.front \
        --out "$OUT_DIR" \
        --label "$LABEL"
done

echo "==[ done. results: $OUT_ROOT ]=="
