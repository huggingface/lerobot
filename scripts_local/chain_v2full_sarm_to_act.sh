#!/usr/bin/env bash
# Chain post-SARM-v2_full pipeline:
#   1. Eval SARM v2_full on v3_eval
#   2. Relabel dataset w/ new SARM
#   3. Build sarm_progress.parquet
#   4. Launch ACT chunk20 RA-BC on G1 w/ new progress
# Run on DL_A6000 after SARM training completes.
set -euo pipefail
cd ~/github.com/orel/lerobot/lerobot
LOG=logs/chain_v2full.log
exec >>"$LOG" 2>&1

echo "=== $(date -u) starting chain ==="

CKPT=outputs/sim_3stage_sarm_v2_full_nostale_2cam/checkpoints/014000/pretrained_model
[[ -d $CKPT ]] || { echo "MISSING ckpt $CKPT"; exit 1; }

# 1. Eval SARM on v3_eval (G0 freed by training)
echo "=== eval SARM ==="
CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run python scripts_local/sarm_full_traj_eval.py \
    --ckpt "$CKPT" \
    --stats local/sim_3stage_v2_full_nostale \
    --full-ds domrachev03/sim_3stage_v3_eval \
    --task 'Three-stage assembly' \
    --image-key observation.images.wrist \
    --out outputs/sarm_eval_v2_full_14k \
    --eps 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
cat outputs/sarm_eval_v2_full_14k/summary.md || true

# 2. Relabel
echo "=== relabel v2_full_nostale w/ new SARM ==="
CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run lerobot-relabel-sarm \
    --src-repo-id local/sim_3stage_v2_full_nostale \
    --sarm-checkpoint "$CKPT" \
    --reward-mode delta \
    --new-repo-id local/sim_3stage_v2_full_nostale_sarm_delta_v2 \
    --task "Three-stage assembly" \
    --head-mode sparse \
    --type sarm_ext \
    --stats local/sim_3stage_v2_full_nostale

# 3. Build progress parquet
echo "=== build sarm_progress.parquet ==="
~/.local/bin/uv run python -m lerobot.scripts.build_rabc_progress_from_delta \
    --src-repo-id local/sim_3stage_v2_full_nostale_sarm_delta_v2 \
    --head-mode sparse \
    --output ~/.cache/huggingface/lerobot/local/sim_3stage_v2_full_nostale_sarm_delta_v2/sarm_progress.parquet

# 4. Launch ACT chunk20 RABC on G1
echo "=== launch ACT G1 ==="
CUDA_VISIBLE_DEVICES=1 nohup ~/.local/bin/uv run lerobot-train \
    --config_path=src/lerobot/rl/act_v2_full_rabc_chunk20_v2sarm_train.json \
    > logs/act_v2_full_rabc_chunk20_v2sarm.log 2>&1 < /dev/null &
ACTPID=$!
echo "ACT PID=$ACTPID"

echo "=== $(date -u) chain done ==="
