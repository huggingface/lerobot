#!/usr/bin/env bash
# A1 整条链 smoke 测(各阶段 60 步): ① fast_pretrain_only → 提取VLM → ② 冻结训expert → eval 1 task
# 验证新代码路径(fast_pretrain_only forward / extract_vlm / 阶段②加载本地VLM / A1 checkpoint 推理)通不通。
set -u
cd /home/anker/projects/lerobot
RUN_ENV=(MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
OL=outputs/a1_smoke.log
log(){ echo "[$(date '+%T')] $*" >> "$OL"; }
log "================ A1 SMOKE START ================"

# ① fast_pretrain_only 60 步
log "stage1 fast_pretrain_only 60 steps ..."
env "${RUN_ENV[@]}" uv run lerobot-train --policy.type=smolvla_ki \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=true --policy.keep_full_vlm=true \
  --policy.fast_pretrain_only=true --policy.enable_fast_action_loss=true \
  --policy.knowledge_insulation=false --policy.train_expert_only=false --policy.flow_loss_weight=0.0 \
  --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=HuggingFaceVLA/libero --env.type=libero --env.task=libero_spatial \
  --batch_size=8 --steps=60 --log_freq=10 --eval_freq=100000 --save_freq=60 \
  --wandb.enable=false --output_dir=./outputs/train/a1_smoke_s1 > outputs/a1_smoke_s1.log 2>&1
RC=$?
log "stage1 rc=$RC ; loss: $(grep -aoE 'loss:[0-9.]+' outputs/a1_smoke_s1.log | head -1) -> $(grep -aoE 'loss:[0-9.]+' outputs/a1_smoke_s1.log | tail -1)"
[ -d outputs/train/a1_smoke_s1/checkpoints/last/pretrained_model ] || { log "stage1 NO checkpoint -> ABORT"; touch outputs/A1_SMOKE_DONE; exit 1; }

# 提取 VLM
log "extract VLM ..."
rm -rf /tmp/a1_smoke_vlm
env "${RUN_ENV[@]}" uv run python3 scripts/extract_vlm.py \
  outputs/train/a1_smoke_s1/checkpoints/last/pretrained_model /tmp/a1_smoke_vlm >> "$OL" 2>&1
[ -f /tmp/a1_smoke_vlm/config.json ] && log "extract OK" || { log "extract FAIL -> ABORT"; touch outputs/A1_SMOKE_DONE; exit 1; }

# ② 加载提取的 VLM + 冻结 + 训 fresh expert 60 步
log "stage2 load extracted VLM, train_expert_only 60 steps ..."
env "${RUN_ENV[@]}" uv run lerobot-train --policy.type=smolvla_ki \
  --policy.vlm_model_name=/tmp/a1_smoke_vlm \
  --policy.load_vlm_weights=true --policy.keep_full_vlm=true \
  --policy.fast_pretrain_only=false --policy.enable_fast_action_loss=false \
  --policy.knowledge_insulation=false --policy.train_expert_only=true \
  --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=HuggingFaceVLA/libero --env.type=libero --env.task=libero_spatial \
  --batch_size=8 --steps=60 --log_freq=10 --eval_freq=100000 --save_freq=60 \
  --wandb.enable=false --output_dir=./outputs/train/a1_smoke_s2 > outputs/a1_smoke_s2.log 2>&1
RC=$?
log "stage2 rc=$RC ; loss: $(grep -aoE 'loss:[0-9.]+' outputs/a1_smoke_s2.log | head -1) -> $(grep -aoE 'loss:[0-9.]+' outputs/a1_smoke_s2.log | tail -1)"
[ -d outputs/train/a1_smoke_s2/checkpoints/last/pretrained_model ] || { log "stage2 NO checkpoint -> ABORT"; touch outputs/A1_SMOKE_DONE; exit 1; }

# eval A1 smoke checkpoint 1 task
log "eval A1 smoke ckpt (1 task, 2 ep) ..."
rm -rf /tmp/a1_smoke_eval
env "${RUN_ENV[@]}" uv run lerobot-eval \
  --policy.path=outputs/train/a1_smoke_s2/checkpoints/last/pretrained_model \
  --env.type=libero --env.task=libero_spatial --env.task_ids="[0]" \
  --eval.batch_size=1 --eval.n_episodes=2 --eval.use_async_envs=false --policy.device=cuda \
  '--env.camera_name_mapping={"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}' \
  --output_dir=/tmp/a1_smoke_eval > outputs/a1_smoke_eval.log 2>&1
[ -f /tmp/a1_smoke_eval/eval_info.json ] && log "eval OK ✅ — A1 整条链通!" || log "eval FAIL ❌ (see outputs/a1_smoke_eval.log)"
log "================ A1 SMOKE DONE ================"
touch outputs/A1_SMOKE_DONE
