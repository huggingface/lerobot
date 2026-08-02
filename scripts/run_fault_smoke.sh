#!/usr/bin/env bash
# Example A/B smoke: same eval twice — once clean, once with action_hold.
#
# "A/B" here only means: compare baseline (A) vs faulted (B) on the same task.
# The fault layer itself is model-agnostic; this script just picks one convenient
# example policy/env. Override them freely:
#
#   POLICY_PATH=lerobot/pi0_... ENV_TYPE=libero ENV_TASK=libero_object \
#     bash scripts/run_fault_smoke.sh injected
#
# Prerequisites: LeRobot + the chosen env extras, and a runnable policy checkpoint.
#
# Usage:
#   bash scripts/run_fault_smoke.sh baseline
#   bash scripts/run_fault_smoke.sh injected
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-baseline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON="${PYTHON:-python}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_libero}"
ENV_TYPE="${ENV_TYPE:-libero}"
ENV_TASK="${ENV_TASK:-libero_object}"

COMMON=(
  --policy.path="${POLICY_PATH}"
  --env.type="${ENV_TYPE}"
  --env.task="${ENV_TASK}"
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
    echo "Video: outputs/eval/fault_smoke_baseline/videos/${ENV_TASK}_0/eval_episode_0.mp4"
    echo "JSON:  outputs/eval/fault_smoke_baseline/eval_info.json"
    ;;
  injected)
    # trigger_step=20: early enough for short successful episodes (~tens of steps)
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
    echo "Video: outputs/eval/fault_smoke_injected/videos/${ENV_TASK}_0/eval_episode_0.mp4"
    echo "JSON:  outputs/eval/fault_smoke_injected/eval_info.json"
    echo "Fault: outputs/eval/fault_smoke_injected/fault_events.jsonl"
    ;;
  *)
    echo "Usage: $0 {baseline|injected}" >&2
    exit 1
    ;;
esac
