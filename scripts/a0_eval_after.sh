#!/usr/bin/env bash
# 接管编排被杀后的剩余工作:等 A0 训练进程(PID 1302992)结束 -> eval A0 -> 汇总 A2 vs A0
# 纯 /proc 文件检测等待(不用 kill/pkill,规避本环境的 exit 144)
set -u
cd /home/anker/projects/lerobot

A0_PID=1302992
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

log "==== a0_eval_after START (waiting A0 PID $A0_PID) ===="
# 1. 等 A0 训练进程结束
while [ -d "/proc/$A0_PID" ]; do sleep 60; done
log "A0 training process $A0_PID gone"
sleep 30

A0CK=outputs/train/smolvla_ki_A0_30k/checkpoints/last/pretrained_model
if [ ! -d "$A0CK" ]; then
  log "ERROR: A0 checkpoint missing at $A0CK -> cannot eval A0"
  touch outputs/OVERNIGHT_DONE
  exit 1
fi
log "A0 checkpoint present: $A0CK"

# 2. eval A0 (per-task, 修正后的相机参数, 带单次 retry)
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
OUTROOT=outputs/eval/A0_30k_libero_spatial
mkdir -p "$OUTROOT"
log "A0 EVAL start -> $OUTROOT"
okc=0
for tid in 0 1 2 3 4 5 6 7 8 9; do
  out="$OUTROOT/task${tid}"
  rm -rf "$out" "$out.log"
  eval_one "$A0CK" "$out" libero_spatial "$tid"
  if [ ! -f "$out/eval_info.json" ]; then
    log "  A0 task$tid retry"; rm -rf "$out" "$out.log"; eval_one "$A0CK" "$out" libero_spatial "$tid"
  fi
  if [ -f "$out/eval_info.json" ]; then okc=$((okc+1)); log "  A0 task$tid OK"; else log "  A0 task$tid FAIL"; fi
done
log "A0 EVAL done okc=$okc/10"

# 3. 汇总 A2 vs A0
log "==================== SUMMARY A2 vs A0 ===================="
uv run python3 - >> "$OL" 2>&1 <<'PY' || log "summary error"
import json, glob, os, statistics
res={}
for tag, root in [("A2 (KI co-train)", "outputs/eval/A2_30k_libero_spatial"),
                  ("A0 (frozen VLM)",  "outputs/eval/A0_30k_libero_spatial")]:
    rows=[]
    for f in sorted(glob.glob(os.path.join(root,"task*","eval_info.json"))):
        try: d=json.load(open(f))
        except Exception: continue
        rows.append((os.path.basename(os.path.dirname(f)), d["overall"]["pc_success"]))
    if rows:
        m=statistics.mean(r[1] for r in rows); res[tag]=m
        print(f"[{tag}] mean success = {m:.1f}%  ({len(rows)} tasks)")
        print("        " + "  ".join(f"{t}:{p:.0f}%" for t,p in rows))
    else:
        print(f"[{tag}] no eval_info.json under {root}")
if "A2 (KI co-train)" in res and "A0 (frozen VLM)" in res:
    d = res["A2 (KI co-train)"] - res["A0 (frozen VLM)"]
    verdict = "KI 有效(A2>A0)" if d>0 else ("KI 无明显增益" if abs(d)<3 else "KI 反而更差(A2<A0)")
    print(f">>> A2 - A0 = {d:+.1f} 个百分点  ->  {verdict}")
PY
log "==== a0_eval_after DONE ===="
touch outputs/OVERNIGHT_DONE
