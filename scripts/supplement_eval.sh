#!/usr/bin/env bash
# 补齐 A2 + A0 在剩余 3 个 suite (object/goal/10) 的 eval。spatial 已有,自动跳过。
# per-task 隔离(规避 EGL 多 context 崩溃),正确相机映射(image/image2, 无 empty_cameras)。
set -u
cd /home/anker/projects/lerobot

COMMON_ENV=(
  HF_HUB_OFFLINE=1
  MUJOCO_GL=egl
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)
OL=outputs/supplement_eval.log
log(){ echo "[$(date '+%F %T')] $*" >> "$OL"; }

eval_one(){  # $1 ckpt  $2 out  $3 suite  $4 tid
  timeout 1200 env "${COMMON_ENV[@]}" uv run lerobot-eval \
    --policy.path="$1" \
    --env.type=libero --env.task="$3" --env.task_ids="[$4]" \
    --eval.batch_size=1 --eval.n_episodes=10 --eval.use_async_envs=false \
    --policy.device=cuda \
    '--env.camera_name_mapping={"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}' \
    --env.max_parallel_tasks=1 \
    --output_dir="$2" > "$2.log" 2>&1
}

eval_suite(){  # $1 ckpt  $2 tag  $3 suite
  local ckpt="$1" tag="$2" suite="$3"
  local root="outputs/eval/${tag}_${suite}"
  mkdir -p "$root"
  log "EVAL $tag $suite start"
  for tid in 0 1 2 3 4 5 6 7 8 9; do
    local out="$root/task${tid}"
    if [ -f "$out/eval_info.json" ]; then log "  $tag $suite t$tid already done"; continue; fi
    rm -rf "$out" "$out.log"
    eval_one "$ckpt" "$out" "$suite" "$tid"
    [ -f "$out/eval_info.json" ] || { rm -rf "$out" "$out.log"; eval_one "$ckpt" "$out" "$suite" "$tid"; }
    [ -f "$out/eval_info.json" ] && log "  $tag $suite t$tid OK" || log "  $tag $suite t$tid FAIL"
  done
  log "EVAL $tag $suite done"
}

A2CK=outputs/train/smolvla_ki_A2_30k/checkpoints/last/pretrained_model
A0CK=outputs/train/smolvla_ki_A0_30k/checkpoints/last/pretrained_model

log "==== supplement eval START (A2+A0 x object/goal/10) ===="
for suite in libero_object libero_goal libero_10; do
  eval_suite "$A2CK" A2_30k "$suite"
  eval_suite "$A0CK" A0_30k "$suite"
done

# 汇总全 4 suite
log "==================== FULL 4-SUITE SUMMARY ===================="
uv run python3 - >> "$OL" 2>&1 <<'PY' || log "summary error"
import json, glob, os, statistics
SUITES=["libero_spatial","libero_object","libero_goal","libero_10"]
for tag in ["A2_30k","A0_30k"]:
    print(f"=== {tag} ===")
    allv=[]
    for s in SUITES:
        vs=[]
        for f in sorted(glob.glob(f"outputs/eval/{tag}_{s}/task*/eval_info.json")):
            try: vs.append(json.load(open(f))["overall"]["pc_success"])
            except Exception: pass
        if vs:
            m=statistics.mean(vs); allv+=vs
            print(f"  {s:<16} {m:5.1f}%  ({len(vs)} tasks)")
        else:
            print(f"  {s:<16} (none)")
    if allv: print(f"  {'OVERALL':<16} {statistics.mean(allv):5.1f}%  ({len(allv)} tasks)")
PY
log "==== supplement eval DONE ===="
touch outputs/SUPPLEMENT_EVAL_DONE
