#!/usr/bin/env bash
# 今晚无人值守编排:等 A2 训练结束 -> eval A2 -> 训练 A0 30k -> eval A0 -> 汇总
# 设计要点:
#  - 串行单 GPU,各阶段顺序执行(前一阶段 wait 完才下一阶段),不并发抢显存
#  - eval 用 per-task 隔离(每 task 单进程),规避 LIBERO/EGL 多 context 崩溃
#  - eval 有 timeout + 单次 retry + "前2个 task 全失败就早停"保护,bug 不拖垮整晚
#  - eval 失败不阻塞 A0 训练;每阶段详细日志写 outputs/overnight_orchestrator.log
set -u
cd /home/anker/projects/lerobot

COMMON_ENV=(
  ALL_PROXY=socks5h://127.0.0.1:1080
  HTTPS_PROXY=socks5h://127.0.0.1:1080
  HTTP_PROXY=socks5h://127.0.0.1:1080
  MUJOCO_GL=egl
  HF_HUB_DOWNLOAD_TIMEOUT=300
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

OL=outputs/overnight_orchestrator.log
log(){ echo "[$(date '+%F %T')] $*" >> "$OL"; }

log "================ overnight orchestrator START ================"

# ---------- helpers ----------
wait_proc_gone(){  # $1 = pgrep -f pattern
  while pgrep -f "$1" >/dev/null 2>&1; do sleep 120; done
}

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

eval_ckpt(){  # $1 ckpt  $2 outroot  $3 suite
  local ckpt="$1" outroot="$2" suite="${3:-libero_spatial}"
  mkdir -p "$outroot"
  log "EVAL start ckpt=$ckpt suite=$suite -> $outroot"
  local okc=0
  for tid in 0 1 2 3 4 5 6 7 8 9; do
    local out="$outroot/task${tid}"
    rm -rf "$out" "$out.log"
    eval_one "$ckpt" "$out" "$suite" "$tid"
    if [ ! -f "$out/eval_info.json" ]; then
      log "  task$tid first try failed -> retry"
      rm -rf "$out" "$out.log"; eval_one "$ckpt" "$out" "$suite" "$tid"
    fi
    if [ -f "$out/eval_info.json" ]; then
      okc=$((okc+1)); log "  task$tid OK"
    else
      log "  task$tid FAIL (see $out.log)"
    fi
    # 系统性失败早停:跑完前 2 个 task 一个都没成,基本是 inference bug,放弃整轮 eval
    if [ "$tid" -eq 1 ] && [ "$okc" -eq 0 ]; then
      log "  ABORT eval: first 2 tasks all failed (likely systemic inference bug); skip rest"
      return 1
    fi
  done
  log "EVAL done okc=$okc/10 -> $outroot"
  return 0
}

# ---------- 1. 等 A2 训练结束 ----------
log "WAIT A2 training (pattern: lerobot-train .* smolvla_ki_A2_30k) to finish ..."
wait_proc_gone "lerobot-train.*smolvla_ki_A2_30k"
sleep 20
A2CK=outputs/train/smolvla_ki_A2_30k/checkpoints/last/pretrained_model
if [ -d "$A2CK" ]; then
  log "A2 training finished; checkpoint present: $A2CK"
  # ---------- 2. eval A2 ----------
  eval_ckpt "$A2CK" outputs/eval/A2_30k_libero_spatial libero_spatial || log "A2 eval aborted/failed (non-fatal, continue to A0)"
else
  log "ERROR: A2 checkpoint missing at $A2CK -> skip A2 eval"
fi

# ---------- 3. 训练 A0 (30k, 冻结 VLM + 仅 flow loss, 同架构) ----------
log "A0 training START (30k steps)"
rm -rf outputs/train/smolvla_ki_A0_30k
env "${COMMON_ENV[@]}" uv run lerobot-train \
  --policy.type=smolvla_ki \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=true --policy.keep_full_vlm=true \
  --policy.knowledge_insulation=false --policy.enable_fast_action_loss=false \
  --policy.train_expert_only=true --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=HuggingFaceVLA/libero --env.type=libero --env.task=libero_spatial \
  --batch_size=16 --steps=30000 --log_freq=100 \
  --eval_freq=100000 --save_freq=5000 --wandb.enable=false \
  --output_dir=./outputs/train/smolvla_ki_A0_30k > outputs/train_A0_30k.log 2>&1
A0RC=$?
log "A0 training EXIT rc=$A0RC"

# ---------- 4. eval A0 ----------
A0CK=outputs/train/smolvla_ki_A0_30k/checkpoints/last/pretrained_model
if [ -d "$A0CK" ]; then
  eval_ckpt "$A0CK" outputs/eval/A0_30k_libero_spatial libero_spatial || log "A0 eval aborted/failed"
else
  log "ERROR: A0 checkpoint missing (train rc=$A0RC) -> skip A0 eval"
fi

# ---------- 5. 汇总 ----------
log "==================== SUMMARY ===================="
uv run python3 - >> "$OL" 2>&1 <<'PY' || log "summary script error"
import json, glob, os
for tag, root in [("A2 (KI co-train)", "outputs/eval/A2_30k_libero_spatial"),
                  ("A0 (frozen VLM)",  "outputs/eval/A0_30k_libero_spatial")]:
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "task*", "eval_info.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ov = d.get("overall", {})
        rows.append((os.path.basename(os.path.dirname(f)), ov.get("pc_success"), ov.get("n_episodes")))
    good = [r[1] for r in rows if r[1] is not None]
    if good:
        mean = sum(good) / len(good)
        print(f"[{tag}] libero_spatial mean success = {mean:.1f}%  ({len(good)}/10 tasks)")
        for name, pc, n in rows:
            print(f"      {name}: {pc}%  (n={n})")
    else:
        print(f"[{tag}] no eval_info.json found under {root}")
PY

log "================ overnight orchestrator DONE ================"
touch outputs/OVERNIGHT_DONE
