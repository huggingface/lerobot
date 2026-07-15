#!/usr/bin/env bash
# Matched SmolVLA ablation for the 10 fps lerobot/robomme dataset.
# Run from the LeRobot repository root on a CUDA machine.

set -euo pipefail

BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
# The published training split has 476,857 execution frames. By default, train on one
# execution-frame epoch; set STEPS explicitly to use a different optimizer-step budget.
TARGET_SAMPLES="${TARGET_SAMPLES:-476857}"
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NUM_PROCESSES))
STEPS="${STEPS:-$(((TARGET_SAMPLES + EFFECTIVE_BATCH_SIZE - 1) / EFFECTIVE_BATCH_SIZE))}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-$(((STEPS + 29) / 30))}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-${STEPS}}"
SEED="${SEED:-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/robomme-smolvla-mem-ablation}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
VARIANT="${VARIANT:-both}"
TASKS="BinFill,PickXtimes,SwingXtimes,StopCube,VideoUnmask,VideoUnmaskSwap,ButtonUnmask,ButtonUnmaskSwap,PickHighlight,VideoRepick,VideoPlaceButton,VideoPlaceOrder,MoveCube,InsertPeg,PatternLock,RouteStick"
TASK_IDS="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]"

case "${VARIANT}" in
  baseline) VARIANTS=(baseline) ;;
  visual-memory) VARIANTS=(visual-memory) ;;
  both) VARIANTS=(baseline visual-memory) ;;
  *) echo "VARIANT must be baseline, visual-memory, or both" >&2; exit 2 ;;
esac

TRAIN_COMMAND=()
if [[ "${RUN_TRAIN}" == "true" ]]; then
  if ((NUM_PROCESSES > 1)); then
    TRAIN_ENTRYPOINT="$(uv run which lerobot-train)"
    TRAIN_COMMAND=(
      uv run accelerate launch
      --multi_gpu
      --num_processes="${NUM_PROCESSES}"
      --mixed_precision=bf16
      "${TRAIN_ENTRYPOINT}"
    )
  else
    TRAIN_COMMAND=(uv run lerobot-train)
  fi
fi

echo "Training ${VARIANT}: ${TARGET_SAMPLES} target samples, global batch ${EFFECTIVE_BATCH_SIZE}, ${STEPS} optimizer steps"

COMMON_TRAIN_ARGS=(
  --policy.path=lerobot/smolvla_base
  --policy.device=cuda
  --policy.push_to_hub=false
  --policy.empty_cameras=1
  --policy.freeze_vision_encoder=false
  --policy.train_expert_only=false
  --dataset.repo_id=lerobot/robomme
  --dataset.training_target_start_feature=exec_start_idx
  '--rename_map={"image":"observation.images.camera1","wrist_image":"observation.images.camera2","state":"observation.state","actions":"action"}'
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}"
  --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}"
  --seed="${SEED}"
  --env_eval_freq=0
  --save_freq=5000
  --wandb.enable="${WANDB_ENABLE}"
)

if [[ "${RUN_TRAIN}" == "true" ]]; then
  for variant in "${VARIANTS[@]}"; do
    MEMORY_ARGS=(--policy.use_visual_memory=false)
    if [[ "${variant}" == "visual-memory" ]]; then
      MEMORY_ARGS=(
        --policy.use_visual_memory=true
        --policy.visual_memory_frames=6
        --policy.visual_memory_stride=10
        --policy.visual_memory_temporal_attention_every=4
      )
    fi
    "${TRAIN_COMMAND[@]}" \
      "${COMMON_TRAIN_ARGS[@]}" \
      "${MEMORY_ARGS[@]}" \
      --output_dir="${OUTPUT_ROOT}/${variant}" \
      --job_name="robomme-smolvla-${variant}"
  done
fi

if [[ "${RUN_EVAL}" == "true" ]]; then
  for variant in "${VARIANTS[@]}"; do
    uv run lerobot-eval \
      --policy.path="${OUTPUT_ROOT}/${variant}/checkpoints/last/pretrained_model" \
      --env.type=robomme \
      --env.task="${TASKS}" \
      --env.dataset_split=test \
      --env.task_ids="${TASK_IDS}" \
      '--rename_map={"observation.images.image":"observation.images.camera1","observation.images.wrist_image":"observation.images.camera2"}' \
      --eval.batch_size=1 \
      --eval.n_episodes=1 \
      --seed="${SEED}" \
      --output_dir="${OUTPUT_ROOT}/eval-${variant}"
  done
fi

if [[ "${RUN_EVAL}" == "true" ]]; then
  uv run python - "${OUTPUT_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for variant in ("baseline", "visual-memory"):
    result_path = root / f"eval-{variant}" / "eval_info.json"
    if not result_path.exists():
        continue
    with result_path.open() as handle:
        info = json.load(handle)
    overall = info["overall"]
    print(
        f"{variant}: success={overall['pc_success']:.2f}% "
        f"avg_reward={overall['avg_sum_reward']:.4f}"
    )
    for task, metrics in info["per_group"].items():
        print(
            f"  {task}: success={metrics['pc_success']:.2f}% "
            f"avg_reward={metrics['avg_sum_reward']:.4f}"
        )
PY
fi
