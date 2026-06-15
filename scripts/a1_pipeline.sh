#!/usr/bin/env bash
# A1 两阶段全流程:① FAST 预训练 VLM(30k)→ 提取 VLM → ② 加载VLM+冻结+训fresh expert(30k)
# → A1 全 4 suite eval → 汇总 A0/A2/A1 三臂。
# 不在此 wait;由调用者保证 GPU 空闲。
# 不显式设 proxy/offline —— 继承环境里的 mihomo 代理(bashrc 已配)走 online。
set -u
cd /home/anker/projects/lerobot

COMMON_ENV=(
  MUJOCO_GL=egl
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)
OL=outputs/a1_pipeline.log
log(){ echo "[$(date '+%F %T')] $*" >> "$OL"; }

S1DIR=outputs/train/smolvla_ki_A1_stage1
VLMDIR=outputs/a1_pretrained_vlm
S2DIR=outputs/train/smolvla_ki_A1_stage2
STEPS="${A1_STEPS:-30000}"

log "================ A1 pipeline START (steps=$STEPS) ================"

# ── 阶段① FAST 预训练 VLM(fast_pretrain_only: 只 FAST loss 训 VLM,跳过 flow expert)──
log "STAGE-1 FAST-pretrain VLM start"
rm -rf "$S1DIR"
env "${COMMON_ENV[@]}" uv run lerobot-train \
  --policy.type=smolvla_ki \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=true --policy.keep_full_vlm=true \
  --policy.fast_pretrain_only=true --policy.enable_fast_action_loss=true \
  --policy.knowledge_insulation=false --policy.train_expert_only=false \
  --policy.flow_loss_weight=0.0 \
  --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=HuggingFaceVLA/libero --env.type=libero --env.task=libero_spatial \
  --batch_size=16 --steps="$STEPS" --log_freq=100 --eval_freq=100000 --save_freq=10000 \
  --wandb.enable=false --output_dir="./$S1DIR" > outputs/train_A1_stage1.log 2>&1
log "STAGE-1 done rc=$?"

S1CK="$S1DIR/checkpoints/last/pretrained_model"
if [ ! -d "$S1CK" ]; then log "ERROR: stage-1 checkpoint missing -> abort"; exit 1; fi

# ── 提取 FAST 预训练的 VLM 为 HF 格式 ──
log "EXTRACT VLM start"
rm -rf "$VLMDIR"
env "${COMMON_ENV[@]}" uv run python3 scripts/extract_vlm.py "$S1CK" "$VLMDIR" >> "$OL" 2>&1
if [ ! -f "$VLMDIR/config.json" ]; then log "ERROR: VLM extraction failed -> abort"; exit 1; fi
log "EXTRACT VLM done -> $VLMDIR"

# ── 阶段② 冻结预训练 VLM + 训 fresh expert ──
log "STAGE-2 train fresh expert on frozen pretrained VLM start"
rm -rf "$S2DIR"
env "${COMMON_ENV[@]}" uv run lerobot-train \
  --policy.type=smolvla_ki \
  --policy.vlm_model_name="$VLMDIR" \
  --policy.load_vlm_weights=true --policy.keep_full_vlm=true \
  --policy.fast_pretrain_only=false --policy.enable_fast_action_loss=false \
  --policy.knowledge_insulation=false --policy.train_expert_only=true \
  --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=HuggingFaceVLA/libero --env.type=libero --env.task=libero_spatial \
  --batch_size=16 --steps="$STEPS" --log_freq=100 --eval_freq=100000 --save_freq=10000 \
  --wandb.enable=false --output_dir="./$S2DIR" > outputs/train_A1_stage2.log 2>&1
log "STAGE-2 done rc=$?"

A1CK="$S2DIR/checkpoints/last/pretrained_model"
if [ ! -d "$A1CK" ]; then log "ERROR: stage-2 checkpoint missing -> abort"; exit 1; fi

# ── A1 全 4 suite eval ──
eval_one(){  # ckpt out suite tid
  timeout 1200 env "${COMMON_ENV[@]}" uv run lerobot-eval \
    --policy.path="$1" --env.type=libero --env.task="$3" --env.task_ids="[$4]" \
    --eval.batch_size=1 --eval.n_episodes=10 --eval.use_async_envs=false --policy.device=cuda \
    '--env.camera_name_mapping={"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}' \
    --env.max_parallel_tasks=1 --output_dir="$2" > "$2.log" 2>&1
}
for suite in libero_spatial libero_object libero_goal libero_10; do
  root="outputs/eval/A1_30k_${suite}"; mkdir -p "$root"
  log "A1 EVAL $suite start"
  for tid in 0 1 2 3 4 5 6 7 8 9; do
    out="$root/task${tid}"
    [ -f "$out/eval_info.json" ] && continue
    rm -rf "$out" "$out.log"; eval_one "$A1CK" "$out" "$suite" "$tid"
    [ -f "$out/eval_info.json" ] || { rm -rf "$out" "$out.log"; eval_one "$A1CK" "$out" "$suite" "$tid"; }
    [ -f "$out/eval_info.json" ] && log "  A1 $suite t$tid OK" || log "  A1 $suite t$tid FAIL"
  done
done

# ── 汇总 A0/A2/A1 三臂全 suite ──
log "==================== 3-ARM SUMMARY (A0/A2/A1) ===================="
uv run python3 - >> "$OL" 2>&1 <<'PY' || log "summary error"
import json, glob, os, statistics
SUITES=["libero_spatial","libero_object","libero_goal","libero_10"]
ARMS=[("A0 frozen","A0_30k"),("A2 KI","A2_30k"),("A1 pretrain+freeze","A1_30k")]
print(f"{'suite':<16}" + "".join(f"{a[0]:>20}" for a in ARMS))
overall={a[1]:[] for a in ARMS}
for s in SUITES:
    row=f"{s:<16}"
    for _,tag in ARMS:
        vs=[json.load(open(f))["overall"]["pc_success"]
            for f in sorted(glob.glob(f"outputs/eval/{tag}_{s}/task*/eval_info.json"))]
        overall[tag]+=vs
        row+=f"{(statistics.mean(vs) if vs else float('nan')):>19.1f}%"
    print(row)
print("-"*76)
row=f"{'OVERALL':<16}"
for _,tag in ARMS:
    vs=overall[tag]; row+=f"{(statistics.mean(vs) if vs else float('nan')):>19.1f}%"
print(row)
PY
log "================ A1 pipeline DONE ================"
touch outputs/A1_DONE
