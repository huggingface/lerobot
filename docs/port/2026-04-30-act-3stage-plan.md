# ACT 3-stage train plan (epic lerobot-75)

start 2026-04-30 evening. SARM iter8 shipped as production reward model.

## context (from epic-52 lessons, 2-stage)

- chunked policy beats single-step BC (BC capped at 6%, ACT v9 hit 85% @ thr=0.90).
- v9 ceiling at thr=0.95 was gripper-coord stale; v11 fixed via tail-30 destale → 40% @ thr=0.95.
- v11 recipe: chunk=10, n_obs_steps=1, MIN_MAX state+action, MEAN_STD visual (imagenet), 80k steps, ResNet18, VAE on, kl=10, dim_model=512.
- temporal_ensemble HURTS gripper sharp transitions; use pure chunked exec at eval.

## what's different now

| | 2-stage (v11) | 3-stage (v1) |
|---|---|---|
| ds | local/sim_assemble_actdp_combined_destale_tail30 (76 eps) | domrachev03/sim_assemble_sarm_multistage_three_stages_success (100 eps) |
| state | 15-D (joints+grip+ee_pose) | 7-D (6 joints + 1 gripper width) |
| action | 5-D (3 ee + 1 yaw + 1 grip discrete) | 5-D (3 ee + 1 yaw + 1 grip continuous in [-1,1]) |
| stages | 4 | 6 |
| reward model | sarm_ext iter5 + merged_v1 | sarm_ext iter8 + sim_3stage_v2_train_fs |
| eval task | "Two-stage assembly" | "Three-stage assembly" |

state/action shapes are auto-handled by ACT (computed from dataset stats); cfg unchanged.

## v1 cfg (epic-75 T2)

`src/lerobot/rl/act_3stage_v2_train.json` — copy of act_v11 with:
- repo_id → `domrachev03/sim_assemble_sarm_multistage_three_stages_success`
- output_dir → `outputs/act_3stage_v2`
- wandb.enable=false (local-only)
- everything else identical to v11

## eval (epic-75 T3)

env cfg `src/lerobot/rl/sim_3stage_act_eval_env.json`:
- env: sim_assembling, AssembleBase-v0, fps=20
- gripper: use_gripper=true, record_gripper_width=true (matches train ds)
- stage_names: 6 stages
- reward_model: sarm_ext, iter8 ckpt, sim_3stage_v2_train_fs stats, task="Three-stage assembly", reward_mode=delta, success_threshold=0.95
- terminate_on_success=true (so DONE = SARM hit)

run:
```
MUJOCO_GL=egl uv run python -m lerobot.scripts.eval_chunk_policy \
    --config_path=src/lerobot/rl/sim_3stage_act_eval_env.json \
    --pretrained=outputs/act_3stage_v2/checkpoints/last/pretrained_model \
    --policy-type=act \
    --task="Three-stage assembly" \
    --n-episodes=20
```

target ≥40% (8/20). gate = SARM thr=0.95 fired before time limit truncation.

## hyp ranking (a priori)

| # | hyp | leverage | iter |
|---|---|---|---|
| 1 | v11 recipe transfers cleanly to 3-stage | hi | v1 |
| 2 | 6-stage horizon longer → may need chunk=20 | mid | v2 if v1 <40% |
| 3 | continuous gripper smoother → less stale, no destale needed | hi | testing in v1 |
| 4 | new state space cleaner → faster convergence | mid | observe loss curves |
| 5 | longer training (120k) helps | low | v3 |

## kill conditions

- ≥40% at thr=0.95 → ship as production ACT for downstream HIL-SERL
- 5 iters fruitless → escalate (encoder swap, frame stack, augs, recollect data)

## results

### v1 (chunk=10, 80k steps) — FAIL

train: 30:25 wall-clock, 80k steps. eval: 0/20 = **0%** at thr=0.95.

per-ep max_step_r distribution: most 0.55-0.58, two outliers ep6=0.723, ep12=0.878. all eps hit 60s timeout; never crossed 0.95.

failure mode: stuck at ~0.55 plateau ≈ stage 3-4 boundary (place_target / approach_cover). same boundary that capped SARM iter8 succ_term at 0.10.

**diagnosis**: BOTH ACT and SARM contribute:
- ACT may genuinely fail late stages (cover placement)
- SARM iter8 hits 0.95 only 10% of GT successes (max≥0.95 rate on full bucket = 0.47, val term≥0.95 = 0.10 — calibration ceiling)
- best ep12 max=0.878 → if SARM were better calibrated, this likely passes thr=0.85

### v3 (chunk=20, 80k steps) — running

hyp: longer chunked exec window reduces replan OOD on 6-stage horizon. 2-stage v9 used chunk=10 successfully but tasks were ~half as long.
