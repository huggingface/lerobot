# 3-Stage Assembly Pipeline (Sim → IL → HIL-SERL)

End-to-end pipeline for the 6-stage MuJoCo assembly task (place_box → place_target → place_cover, with stage-pairs for approach + place). Path:

```
record demos → SARM train → SARM-relabel → ACT train (RA-BC) → optional HIL-SERL residual finetune
```

Production goal: ACT policy reaching ≥95% SARM progress in ≥10% of rollouts and ≥80% reach last stage, IL-only.

---

## 1. Data collection

### 1.1 Sim env

`simulator_for_il_rl/env.py` — MuJoCo `AssembleBase-v0`. Renderer 128×128 native. Cameras: `cam_front` (table-side), `cam_wrist` (gripper-mounted). State: 7-D = 6 joint angles + 1 gripper finger qpos (`left_driver_joint`).

Scene: `scene.xml` (cap rgba 0.5 light gray, cam_front at `(0,-1.1,0.37) fovy 45`).

### 1.2 Recording

```bash
DISPLAY=:0 uv run python -m lerobot.rl.gym_manipulator \
  --config_path=src/lerobot/rl/sim_3stage_multistage_record.json
```

Operator drives via gamepad (`stage_advance_button=0` to commit a stage transition). Six sparse subtask labels:
`approach_box, bring_box, approach_target, place_target_in_the_box, approach_cover, place_cover_on_the_box`.

### 1.3 Full vs partial successes

- **Full success**: episode reaches stage 6 + receives `next.reward ≥ 0.5` (env signals task complete).
- **Partial success**: operator advances stages physically reached, then terminates without completing. `next.reward` stays 0.
- Recordings labeled by *what physically happened*. Operator must NOT press stage-advance for stages not actually reached.

### 1.4 Dataset prep

- HF format via `LeRobotDataset`, parquet + h264 mp4 per cam.
- Helpers (`scripts_local/`):
  - `combine_datasets.py` — concat HF datasets, exclude bad eps
  - `subset_episodes.py` — keep specific ep indices
  - `resize_videos.py` — ffmpeg downscale (224→128)
  - `filter_stale_state_frames.py` — drop idle frames (`||Δstate|| < 5e-3` for ≥3 consec) → `_nostale` variant
  - `write_temporal_proportions.py` — compute `temporal_proportions_{sparse,dense}.json` from sparse subtask durations
  - `build_clip_cache.py` — cache CLIP image features (B/32 default) → `meta/clip_cache.npz` so SARM training skips encoder forward

---

## 2. SARM training

Stage-Aware Reward Model (`lerobot_policy_sarm`, `type="sarm_ext"`). CLIP-frozen vision + transformer aggregator + dual heads (sparse stage classifier + dense progress regressor). Outputs per-frame progress ∈ [0,1] and stage probabilities.

### 2.1 Cfg recipe (production)

```json
{
  "type": "sarm_ext",
  "image_keys": ["observation.images.front", "observation.images.wrist"],
  "n_obs_steps": 8, "frame_gap": 5, "max_rewind_steps": 3,
  "batch_size": 32, "stage_loss_weight": 3.0,
  "annotation_mode": "dual"
}
```

### 2.2 Training launch

Use `scripts_local/sarm_train_no_video.py` (skips video decode at training time → CLIP cache hit). 14k steps × bs=32 ≈ 25 min on A6000.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts_local/sarm_train_no_video.py \
  --config_path=src/lerobot/rl/sim_3stage_sarm_<variant>_train.json
```

### 2.3 Eval — 10 gates

`python -m lerobot_policy_sarm.eval_sarm_sim_assemble`. Multi-bucket val dataset (`sim_3stage_v2_val_fs`: 158 eps, 0/6..5/6 + full). Gates:

| gate | thr | direction |
|---|---|---|
| succ_term_rate | ≥0.95 | full eps' terminal ≥ thr |
| lin_mad | ≤0.25 | full eps deviation from linear |
| mean_mid | ≥0.25 | mid-time prog |
| monotonicity | ≥0.85 | non-decreasing |
| last_stage_max_prog_rate | ≥1.0 | full eps reach last stage |
| fail_term_rate | ≤0.0 | partials don't finish high |
| zero_max_ge_0.5 | ≤0.0 | 0-stage eps stay LOW (no hallucination) |
| plateau_ok_rate | ≥0.8 | partials' peak in plateau zone |
| stage_not_exceed_rate | ≥0.9 | (#1 priority) pred stage ≤ GT stage |
| stage_not_below_rate | ≥0.7 | pred stage ≥ GT stage |

Priority: `stage_not_exceed > linearity > stage_not_below`. Reject if `zero_max_ge_0.5 > 0` (hallucination).

---

## 3. RA-BC (Reward-Aware Behavior Cloning)

Per-frame loss weighting by SARM-derived progress delta. Frames where SARM says "task progressing" get full weight; stale/regressing frames get zero.

### 3.1 Relabel dataset

```bash
uv run lerobot-relabel-sarm \
  --src-repo-id local/<dataset_nostale> \
  --sarm-checkpoint outputs/<sarm_variant>/checkpoints/014000/pretrained_model \
  --reward-mode delta \
  --new-repo-id local/<dataset>_sarm_delta \
  --task "Three-stage assembly" \
  --head-mode sparse --type sarm_ext \
  --stats local/<dataset_nostale>
```

`--reward-mode delta`: writes `next.reward[t] = sarm_progress[t] − sarm_progress[t−1]`. Cumulative sum recovers original progress.

### 3.2 Build progress lookup parquet

```bash
uv run python -m lerobot.scripts.build_rabc_progress_from_delta \
  --src-repo-id local/<dataset>_sarm_delta \
  --head-mode sparse \
  --output ~/.cache/huggingface/lerobot/local/<dataset>_sarm_delta/sarm_progress.parquet
```

Integrates per-ep deltas → `progress[t]`. Lookup at training time; no SARM forward in train loop.

### 3.3 Weight formula

For batch frame `i`:
```
delta_i = progress[i + chunk_size] − progress[i]
soft_w = clip( (delta − (μ − 2σ)) / (4σ + ε), 0, 1 )       # μ,σ over all deltas
w = 1                       if delta > κ      # high-progress
w = soft_w                  if 0 ≤ delta ≤ κ
w = 0                       if delta < 0      # regression — drop
```

κ default 0.01. Then `w *= B / Σw`. Implemented in `lerobot/utils/rabc.py:RABCWeights`.

---

## 4. ACT training (with RA-BC)

Action Chunking Transformer (`lerobot/policies/act`), modified to support per-sample loss for RA-BC weighting.

### 4.1 Cfg recipe (production v11)

```json
{
  "use_rabc": true,
  "rabc_progress_path": "...",
  "rabc_kappa": 0.01,
  "rabc_head_mode": "sparse",
  "policy": {
    "type": "act",
    "n_obs_steps": 1, "chunk_size": 20, "n_action_steps": 20,
    "vision_backbone": "resnet18", "use_vae": true, "kl_weight": 10.0,
    "dim_model": 512, "n_heads": 8, "dim_feedforward": 3200,
    "n_encoder_layers": 4, "n_decoder_layers": 1,
    "optimizer_lr": 1e-5, "device": "cuda", "use_amp": true
  },
  "image_transforms": {"enable": true, "max_num_transforms": 3}
}
```

### 4.2 ACT.forward modifications

`lerobot/policies/act/modeling_act.py:forward(batch, reduction="mean"|"none")`:
- L1 per-sample loss: `mean(chunk × action_dim)`, keep batch dim
- KLD per-sample, summed over latent dim
- `reduction="none"` returns per-sample `(B,)` for RA-BC weighting in `lerobot_train.py:update_policy`:

```python
per_sample_loss, _ = policy.forward(batch, reduction="none")
loss = (per_sample_loss * rabc_weights).sum() / (rabc_weights.sum() + ε)
```

### 4.3 Training launch

```bash
CUDA_VISIBLE_DEVICES=0 uv run lerobot-train \
  --config_path=src/lerobot/rl/act_<variant>_train.json
```

Standard 80k steps × bs=16 ≈ 85 min on A6000. ~16 step/s w/ num_workers=4.

### 4.4 Eval

`lerobot/scripts/eval_chunk_policy` w/ `sim_3stage_act_v4_sarmreward_eval_env.json`. Per-rollout cumulative SARM reward (= final progress). Filename: `epXX_<succ|fail>_lenYYY_bestRZ.ZZ.mp4` (Z = max cumulative reached).

Production criteria (n=20):
- success_threshold = 0.95
- ≥10% rollouts hit success
- ≥80% rollouts reach last stage (max_cum_progress > end-of-stage-5 boundary ~0.633)

---

## 5. HIL-SERL residual fine-tune (optional, post-IL)

Reference path (not used for IL-only goal). After ACT IL is shippable, residual SAC actor adds correction via gamepad intervention.

### 5.1 Setup

`scripts/bc_pretrain_sac.py` — BC-pretrain SAC actor on relabeled dataset using same RA-BC weights. Then `lerobot-rl-actor` + `lerobot-rl-learner` start SAC w/ pretrained actor.

### 5.2 Cfg

`sim_3stage_sarm_teleop_env.json` (or HIL-SERL train cfg, currently dead-end):
- `reward_model.type=sarm_ext` → SARM gives per-step reward in env
- Operator gamepad intervenes during rollouts (`stage_advance_button=0`) — interventions logged as expert demos
- SAC actor learns residual correction Δa added to base ACT action

### 5.3 Training

```
lerobot-rl-actor + lerobot-rl-learner with pretrained SAC actor +
  base_policy=ACT_v11 ckpt + reward=SARM
```

Goal (separate effort): success_rate ≥ 90% w/o human intervention. **NOT pursued for current IL-only target.**

---

## File index

| concern | files |
|---|---|
| Sim env | `simulator_for_IL_RL/simulator_for_il_rl/env.py`, `lerobot/envs/sim_assembling.py`, `lerobot/rl/gym_manipulator.py` |
| Recording | `lerobot/rl/sim_3stage_multistage_record.json` |
| SARM cfg type | `lerobot_policy_sarm/src/lerobot_policy_sarm/configuration_sarm.py`, `processor_sarm.py`, `modeling_sarm.py` |
| SARM eval | `lerobot_policy_sarm/src/lerobot_policy_sarm/eval_sarm_sim_assemble.py` |
| ACT eval | `lerobot/src/lerobot/scripts/eval_chunk_policy.py` |
| RA-BC | `lerobot/utils/rabc.py`, `lerobot/scripts/build_rabc_progress_from_delta.py`, `lerobot/rl/relabel_sarm.py` |
| ACT mod | `lerobot/policies/act/modeling_act.py`, `lerobot/scripts/lerobot_train.py` |
| Dataset prep | `lerobot/scripts_local/{combine_datasets,subset_episodes,resize_videos,filter_stale_state_frames,write_temporal_proportions,build_clip_cache,merge_two_datasets}.py` |
| HIL-SERL | `lerobot/rl/learner.py, actor.py, buffer.py`, `lerobot/policies/sac/{configuration_sac,modeling_sac}.py`, `lerobot/scripts/bc_pretrain_sac.py` |
| Iteration logs | `docs/sarm_iterations.md`, `docs/act_iterations.md`, `docs/sarm_v4_iteration.md` |
