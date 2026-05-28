# Compact handoff — sim_assembling 6-stage (2026-05-15)

## Stack
- Repo: `/home/dom-iva/github.com/orel/lerobot/lerobot` (VAlikV fork), Python 3.12 via `uv`, run `~/.local/bin/uv run --no-sync`.
- Env: `sim_assembling/AssembleBase-v0` MuJoCo (`simulator_for_IL_RL/` editable). 6 sparse stages: approach_box → bring_box → approach_target → place_target_in_the_box → approach_cover → place_cover_on_the_box.
- Obs: `observation.images.front` + `observation.images.wrist` (128×128 uint8), `observation.state` 7-D (6 arm joints + gripper_width, `record_gripper_width=true`).
- Action: 5-D (`delta_x, delta_y, delta_z, delta_yaw, gripper` ∈ [-1,1]). `action_step_size=0.01`, `yaw_step_size=0.1`.
- EE bounds (match recording): min `[-0.1708, -0.8781, 0.33]`, max `[0.18, -0.4845, 0.38]`.
- Object spawn offset `[0.1, 0, 0]` baked in; per-reset jitter ±0.02 m + ±π/4 yaw.
- Teleop: DualSense via pygame (RC10_control port). Triggers = axes (R2=5, L2=4); past bugs all fixed (gripper-on-reset, mode=record, terminate_on_success=false for CNN collection).
- render_mode: `"all"` = passive viewer + offscreen cam render (after patch to `sim_assembling.py` metadata).
- Two GPU pools: local G0 (RTX 4070 Ti Super 16GB, DISPLAY=:1), DL_A6000 `ssh -p 8003 dom_iva@143.248.121.169` (4× A6000 48GB).
- **CRITICAL eval bug fixed (2026-05-12)**: `eval_chunk_policy.py` was missing `make_pre_post_processors` wrap → all pre-fix ACT eval results invalid. Patched: applies preprocessor before `select_action`, postprocessor after.
- Calibrated CNN threshold = **0.99** (TPR=0.98 on demos, FPR=0% on user-confirmed fake-succ rollouts).

## SARM training results
40+ iterations across v2/v3/v4 datasets + shorthor sweep (a/b/c/e/f/g/h/k/l/m/n) yielded champion **K** (`outputs/sarm_shorthor_k_n4g2_sw20/checkpoints/002500`): recipe `n_obs=4, frame_gap=2, sw=20, freeze_clip=False, clip_lr=5e-7, paperfull base, 2.5k steps`, 2cam (front+wrist), 6-stage sparse head. Eval on `domrachev03/sim_3stage_v2_val_fs` w/ stats `local/sim_3stage_v2_full_v2_nostale`: **10/11 sync gates, mean_max=0.961, succ=0.83, lag=0.30s**. Known failure: hallucinates "halfway done" within 1 sec on stuck-arm rollouts (mean stage_conf=0.90, wrong stage) — only reliable on val_fs partial-fail demos. SARM v4 dataset is bottleneck (mean_mid plateau ≈ 0.302 vs v2 repro 0.498-0.565); model-side iteration exhausted.

## ACT training
v11 recipe (chunk=10, VAE on, kl=10, n_obs=1, MIN_MAX state+action, MEAN_STD visual ImageNet, ResNet18, lr=1e-5, batch=16, 80k) is the working backbone. Champion **6-stage BC chunk=10 60k** (`outputs/act_v2_full_6stg_bc_chunk10_v11/checkpoints/060000`): **90% halfway-CNN succ @ thr=0.99** on `local/sim_3stage_v2_full_v2_succonly_destale_tail30` (205 succ eps, tail-30 destale). Curriculum (init from first4 BC c10 → fine-tune full 6-stage @ LR 3e-6, 40k) = **80%**. chunk=20 = 50%. chunk=80 = 0% (open-loop too long for grasp/place). Visual reality: policy approaches box + grasps green block but **does NOT cleanly drop target inside box** (false-positive cases at lower threshold come from "arm near box + holding object"); user accepts "almost place" as success. Stage 6 (place_cover_on_the_box) unsolved — all 6-stage policies plateau at max_cum 0.85-0.95.

## HIL-SERL residual — what to try next
**Goal:** fine-tune chunk=10 60k champion to (a) cleanly complete stage 4 placement, (b) extend to stage 6 cover placement.

**Setup:**
- Cfg: `src/lerobot/rl/sim_residual_K_chunk10_v1_train.json`. Residual SAC over frozen ACT base.
- base_policy_path: `outputs/act_v2_full_6stg_bc_chunk10_v11/checkpoints/060000/pretrained_model`
- residual_action_scale: `[0.1, 0.1, 0.1, 0.1, 1.0]` (xyz/yaw small, gripper=1.0 to flip across deadband)
- freeze_base_policy: true, bc_loss_weight: 0.0 (pure SAC)
- Reward: K SARM, `success_threshold=0.95`, `success_terminal_bonus=5.0`
- SAC params: discount=0.97, critic_lr=actor_lr=3e-4, target_entropy=-5.0, batch_size=256, online_steps=200k, online_step_before_learning=1000, num_critics=2, vision_encoder_name=helper2424/resnet10 (frozen)
- render_mode: `"all"` (viewer + cam obs both); fix landed in `sim_assembling.py` metadata.

**Launch (two terminals, local G0):**
```bash
cd /home/dom-iva/github.com/orel/lerobot/lerobot
# Term 1 — learner first (gRPC :50051)
DISPLAY=:1 CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run --no-sync python -m lerobot.rl.learner \
    --config_path=src/lerobot/rl/sim_residual_K_chunk10_v1_train.json
# Term 2 — actor (mujoco viewer pops on first reset)
DISPLAY=:1 CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run --no-sync python -m lerobot.rl.actor \
    --config_path=src/lerobot/rl/sim_residual_K_chunk10_v1_train.json
```

**Post-train eval (use thr=0.99 on halfway CNN):**
```bash
DISPLAY=:1 CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run --no-sync python -m lerobot.scripts.eval_chunk_policy \
    --config_path=src/lerobot/rl/sim_3stage_act_eval_env_K_sync_fix4_img128.json \
    --pretrained=outputs/residual_K_chunk10_v1/checkpoints/<step>/pretrained_model \
    --task "Three-stage assembly" --policy-type=sac --n-episodes=10 \
    --cnn-ckpt=outputs/cnn_halfway_v1/best.pt --cnn-thr 0.99 \
    --video-dir outputs/residual_K_chunk10_v1_rollouts
```

**Tuning levers if no learning in 10k steps:**
- Lower `success_threshold` 0.95 → 0.85 (chunk10 max_cum 0.94 — rarely hits 0.95).
- Add BC regularization: `bc_loss_weight=0.1` + provide offline `dataset: local/sim_3stage_v2_full_v2_succonly_destale_tail30`.
- If gripper command oscillates: drop `residual_action_scale[4]` 1.0 → 0.5.
- Memory caveat: both procs on G0 share ~10GB VRAM; if OOM, drop `image_encoder_hidden_dim` 32→16.

**Alternative non-HIL-SERL levers (brainstormed but not run):**
1. Strict sim-GT CNN (query target_pos inside box bbox from MuJoCo state, retrain CNN with ground-truth labels).
2. RA-BC w/ halfway CNN as critic (replace SARM critic).
3. Higher resolution chunk=10 (224×224).
4. Auxiliary head: predict P(aligned) per chunk timestep.

## Champion artifacts
```
outputs/sarm_shorthor_k_n4g2_sw20/checkpoints/002500/        SARM K (10/11 gates)
outputs/act_v2_full_6stg_bc_chunk10_v11/checkpoints/060000/  ★ 6-stage champ 90% halfway
outputs/act_v2_full_6stg_bc_curriculum_v11/checkpoints/040000/  curriculum 80%
outputs/act_v2_first4_bc_v11/checkpoints/080000/             first4 baseline (100%)
outputs/cnn_halfway_v1/best.pt                                halfway success CNN (overall 0.958, pos 0.922; use thr 0.99)
outputs/cnn_v2_front_v1/best.pt                                full-success CNN
outputs/sarm_gate_eval_shorthor_k_002500/                     SARM K eval plots
local/sim_3stage_v2_full_v2_succonly_destale_tail30           ACT train ds (205 succ, tail-30 destale)
local/sim_3stage_v2_full_v2_nostale                           SARM stats source
```

## Server state (post-cleanup, 2026-05-15)
Local disk: 473G free (after 254G prune from 264G outputs). Remote DL_A6000: 41G free. Both clean — only relevant ckpts/CNNs/sample videos kept.
