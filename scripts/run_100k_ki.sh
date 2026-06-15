#!/usr/bin/env bash
# 最终阶段:A2 KI 配方 bs16 100k 从头训练(~5.8 epoch,逼近论文)+ 全 4 suite eval。
set -u
cd /home/anker/projects/lerobot

COMMON_ENV=(
  MUJOCO_GL=egl
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)
OL=outputs/run_100k.log
log(){ echo "[$(date '+%F %T')] $*" >> "$OL"; }
STEPS="${KI_STEPS:-100000}"
DIR=outputs/train/smolvla_ki_100k

log "================ 100k KI training START (steps=$STEPS) ================"
rm -rf "$DIR"
env "${COMMON_ENV[@]}" uv run lerobot-train \
  --policy.type=smolvla_ki \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=true --policy.keep_full_vlm=true \
  --policy.knowledge_insulation=true --policy.enable_fast_action_loss=true \
  --policy.train_expert_only=false --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=HuggingFaceVLA/libero --env.type=libero --env.task=libero_spatial \
  --batch_size=16 --num_workers=16 --steps="$STEPS" --policy.scheduler_decay_steps="$STEPS" \
  --log_freq=100 --eval_freq=200000 --save_freq=10000 \
  --wandb.enable=false --output_dir="./$DIR" > outputs/train_100k.log 2>&1
log "100k training done rc=$?"

CK="$DIR/checkpoints/last/pretrained_model"
if [ ! -d "$CK" ]; then log "ERROR: 100k checkpoint missing -> abort"; exit 1; fi

eval_one(){  # ckpt out suite tid
  timeout 1200 env "${COMMON_ENV[@]}" uv run lerobot-eval \
    --policy.path="$1" --env.type=libero --env.task="$3" --env.task_ids="[$4]" \
    --eval.batch_size=1 --eval.n_episodes=10 --eval.use_async_envs=false --policy.device=cuda \
    '--env.camera_name_mapping={"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}' \
    --env.max_parallel_tasks=1 --output_dir="$2" > "$2.log" 2>&1
}
for suite in libero_spatial libero_object libero_goal libero_10; do
  root="outputs/eval/KI100k_${suite}"; mkdir -p "$root"
  log "100k EVAL $suite start"
  for tid in 0 1 2 3 4 5 6 7 8 9; do
    out="$root/task${tid}"; [ -f "$out/eval_info.json" ] && continue
    rm -rf "$out" "$out.log"; eval_one "$CK" "$out" "$suite" "$tid"
    [ -f "$out/eval_info.json" ] || { rm -rf "$out" "$out.log"; eval_one "$CK" "$out" "$suite" "$tid"; }
    [ -f "$out/eval_info.json" ] && log "  100k $suite t$tid OK" || log "  100k $suite t$tid FAIL"
  done
done

log "==================== 100k FULL 4-SUITE SUMMARY ===================="
uv run python3 - >> "$OL" 2>&1 <<'PY' || log "summary error"
import json, glob, os, statistics
SUITES=["libero_spatial","libero_object","libero_goal","libero_10"]
allv=[]
for s in SUITES:
    vs=[json.load(open(f))["overall"]["pc_success"]
        for f in sorted(glob.glob(f"outputs/eval/KI100k_{s}/task*/eval_info.json"))]
    allv+=vs
    print(f"  {s:<16} {(statistics.mean(vs) if vs else float('nan')):5.1f}%  ({len(vs)} tasks)")
if allv: print(f"  {'OVERALL':<16} {statistics.mean(allv):5.1f}%  ({len(allv)} tasks)")
PY
log "================ 100k DONE ================"
touch outputs/RUN100K_DONE
