#!/usr/bin/env bash
# Smoke A/B for ActionHoldFault on LIBERO/Panda (laptop-friendly SmolVLA).
#
# Prerequisites:
#   - LeRobot installed with LIBERO extras and a working GPU/CPU policy env
#   - Cached or downloadable `lerobot/smolvla_libero` weights
#
# Usage:
#   bash scripts/run_fault_smoke.sh baseline
#   bash scripts/run_fault_smoke.sh injected
#
# Optional env overrides:
#   PYTHON=python3.12 MUJOCO_GL=egl HF_HUB_OFFLINE=0 bash scripts/run_fault_smoke.sh injected
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-baseline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON="${PYTHON:-python}"

COMMON=(
  --policy.path=lerobot/smolvla_libero
  --env.type=libero
  --env.task=libero_object
  --env.task_ids="[0]"
  --env.control_mode=relative
  --env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
  --policy.empty_cameras=1
  --eval.batch_size=1
  --eval.n_episodes=1
  --eval.use_async_envs=false
  --env.max_parallel_tasks=1
  --seed=1000
  --policy.device="${POLICY_DEVICE:-cuda}"
)

case "$MODE" in
  baseline)
    "$PYTHON" -m lerobot.scripts.lerobot_eval \
      "${COMMON[@]}" \
      --fault.enabled=false \
      --output_dir=outputs/eval/fault_smoke_baseline \
      --job_name=fault_smoke_baseline
    echo "Video: outputs/eval/fault_smoke_baseline/videos/libero_object_0/eval_episode_0.mp4"
    echo "JSON:  outputs/eval/fault_smoke_baseline/eval_info.json"
    ;;
  injected)
    # trigger_step=20: early enough for short successful SmolVLA episodes (~tens of steps)
    "$PYTHON" -m lerobot.scripts.lerobot_eval \
      "${COMMON[@]}" \
      --fault.enabled=true \
      --fault.type=action_hold \
      --fault.trigger_step=20 \
      --fault.duration=8 \
      --fault.probability=1.0 \
      --fault.seed=42 \
      --fault.log_path=outputs/eval/fault_smoke_injected/fault_events.jsonl \
      --output_dir=outputs/eval/fault_smoke_injected \
      --job_name=fault_smoke_injected
    echo "Video: outputs/eval/fault_smoke_injected/videos/libero_object_0/eval_episode_0.mp4"
    echo "JSON:  outputs/eval/fault_smoke_injected/eval_info.json"
    echo "Fault: outputs/eval/fault_smoke_injected/fault_events.jsonl"
    ;;
  *)
    echo "Usage: $0 {baseline|injected}" >&2
    exit 1
    ;;
esac
