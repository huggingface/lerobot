#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu24/Teleop/lerobot
e=/home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_act_additive_three_model_200k_v1/formal_logs
mkdir -p "$e"
printf 'started_utc=%s\n' "$(date -u +%FT%TZ)" > "$e/sequence.status"
for n in r24_repeat r24_realgap24 r24_localsim24gap; do
  printf 'running=%s utc=%s\n' "$n" "$(date -u +%FT%TZ)" >> "$e/sequence.status"
  .venv/bin/python -m lerobot.scripts.lerobot_train \
    --config_path "experiments/task1_picklift_act_additive_three_model_200k_v1/configs/${n}_full.json" \
    > "$e/${n}.log" 2>&1
  printf 'completed=%s utc=%s\n' "$n" "$(date -u +%FT%TZ)" >> "$e/sequence.status"
done
printf 'status=complete utc=%s\n' "$(date -u +%FT%TZ)" >> "$e/sequence.status"
