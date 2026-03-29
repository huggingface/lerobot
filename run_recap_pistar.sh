#!/usr/bin/env bash
# Run in background with:
#   nohup ./run_recap_pistar.sh > train.log 2>&1 &
#   tail -f train.log
set -euo pipefail

python -m lerobot.rl.algorithms.RECAPTrainPiStar \
  --repo_id="jackvial/so101_pickplace_recap_merged_v2" \
  --output_dir="${HOME}/code/lerobot/outputs/recap_pistar_train_1" \
  --value_network_checkpoint="${HOME}/code/lerobot/outputs/so101_pickplace_recap_value/checkpoints/last.pt" \
  --epochs=10 \
  --batch_size=10 \
  --learning_rate=1e-4 \
  --val_split_ratio=0.1 \
  --validate_every_n_train_steps=1000 \
  --c_fail=500.0 \
  --advantage_threshold=0.0 \
  --advantage_dropout=0.3 \
  --log_every_n_steps=10 \
  --model_precision="bfloat16" \
  --freeze_vision_encoder=true \
  --advantage_cache_path="${HOME}/code/lerobot/outputs/advantage_cache.json" \
  --wandb_project="recap-pistar" \
  --wandb_run_name="pistar-run-gemma-300m-3"
