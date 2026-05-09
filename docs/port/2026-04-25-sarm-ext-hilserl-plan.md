# SARM-ext + 2cam + HIL-SERL 90% — plan

**Epic** lerobot-33. Date 2026-04-25.

## Goal
SARM→ext plugin. +2cam. Retrain merged ds. Tight eval (stage≤cur). HIL-SERL w/ SARM → ≥90% succ.

## Constraints
- No commit lerobot-panda.
- Ext SARM repo: commit OK, no co-auth, no push.
- Lerobot main: no commit/push (user verifies).
- Caveman ultra: terse md + beads.

## Subtasks
| ID | Task | Blocks |
|---|---|---|
| 34 | T1 explore SARM+BYOP+rdp | 35 |
| 35 | T2 port→lerobot_policy_sarm | 36 |
| 36 | T3 2-cam input | 39 |
| 37 | T4 merge ds + stride split | 38,39 |
| 38 | T5 new eval (stage≤cur) | 39 |
| 39 | T6 train 2cam + brainstorm | 40 |
| 40 | T7 iter til gates | 41 |
| 41 | T8 HIL-SERL 90% | — |

## Parallelism
- t0: T1 + T4 (indep)
- t1: T2 after T1; T5 after T4
- t2: T3 after T2; T6 after T3+T4+T5
- t3: T7 → T8

## BYOP pattern (ex lerobot_rdp)
- pkg `lerobot_policy_<name>` (prefix auto-discover)
- `@PreTrainedConfig.register_subclass("<name>")` on cfg
- `PreTrainedPolicy` subclass on model
- exports via `__init__`: `Config`, `Policy`, `make_*_pre_post_processors`
- `pyproject.toml` minimal, hatchling build
- install `uv pip install -e path`

## Datasets
- `domrachev03/sim_assemble_sarm_multistage_two_stages_filtered` (50 ep, ~10.7k fr, bucket 5/5/5/5/30)
- `domrachev03/sim_assemble_sarm_multistage_two_stages_2` (~51 ep, ~14k fr, bucket ~12/10/9/10/10 evolving)
- Split: every-Nth-frame (N≈10) val, rest train.

## Eval metric (new primary)
`stage_not_exceed_rate = frames(pred_stage ≤ gt_current_stage) / total`. Per-ep + agg. Gate ≥ 0.9.

## Gates (combined)
- Old: TP≥95% / FP=0% / lin_mad≤0.25 / mean_mid≥0.25 / monotonicity≥0.85 / plateau_ok ±0.10
- New: stage_not_exceed ≥ 0.9

## HIL-SERL protocol (T8) — detail

### T8a: extend `SARMRewardProcessorStep` multi-cam
- Now: single `_image_key` + single ring buf.
- Multi: detect `model.config.image_keys` → N buffers (dict key→deque).
- `_push_obs_to_buffer`, `_snapshot_buffers`, `_build_window_from_snapshot` op cross-cam.
- Obs consumer: HIL-SERL actor needs reward per-step → each step obs has both image keys → push to each buf.
- Smoke: `python -c "step=SARMRewardProcessorStep(...); step({obs...})"` confirm N-cam path.

### T8b: relabel demo ds
- Demo src: biggest SARM-annot success set. Candidates:
  - `domrachev03/sim_assemble_manual_two_stages` (47 filtered @ local/sim_assemble_manual_filtered).
  - Fallback: 40 full-succ eps ex merged ds.
- Cmd: `lerobot-relabel-sarm --src-repo-id=<demo ds> --sarm-checkpoint=outputs/.../iter5.../last/pretrained_model --reward-mode=delta --task="Two-stage assembly" --stats=local/sim_assemble_sarm_merged_v1 --device=cuda`. Out: `<src>_sarm_delta`.
- Must use `reward_mode=delta` (potential-based shape = SAC-friendly). Same mode MUST match online actor/learner.

### T8c: 15-min smoke
- Env cfg: `src/lerobot/rl/sim_assembling_sarm_env.json` (exist, tuned iter-5 ckpt + task text).
- Two terms: actor + learner via gRPC.
- Save-ckpt: off → avoid mem overflow.
- Wandb: on → see Q-loss/returns trends.

### T8d: triage
- Succ rate, term reward dist, policy entropy, Q value curves.
- Tweak: batch size, discount, entropy coef, lr.

### T8e: 1-h run → 90%
- After smoke OK, run 1h.
- Fallback if short demos: convert 40 full-succ merged eps → add demo set.

### Mem cap strats
- Off `save_checkpoint` mid-run.
- Shrink `replay_buffer_capacity` if OOM.
- Cut `batch_size` if actor→learner backpressure.

## Ideas bank (T6 brainstorm)
- Force feats (proprio + wrench)
- τ-cap
- Rewind sched sweep
- Augs (color jitter, crop)
- stage_loss_weight sweep 3/5/10
- dense vs dual vs single
- obs horizon: 4/8/12
- frame_gap: 3/5/8
- Feat fusion: concat / bilinear / cross-attn (front↔wrist)

## Iter log

### 2026-04-25
- T1 ✓ explore (inline, closed)
- T2 ✓ port to `lerobot_policy_sarm/` (sarm_ext key, SARMPolicy alias, make_sarm_ext_pre_post_processors); 50-step smoke PASS (loss 3.0→2.1). Lerobot edits: `scripts/lerobot_train.py` (type.startswith("sarm_")), `policies/factory.py` (fwd dataset_meta), `processor/reward_model/sarm.py` (dispatch on type=sarm_ext).
- T3 ✓ 2-cam: `image_keys` list cfg, multi-stream CLIP, `num_cameras` wired; 50-step smoke PASS (loss 3.0→2.75, 120M params).
- T4 ✓ merged ds `local/sim_assemble_sarm_merged_v1` (103 ep/25448 fr; buckets 0:18,1:15,2:15,3:15,4:40). Val stride idx json. Merged temporal props frame-weighted.
- T5 ∈ eval metric `stage_not_exceed_rate` added (gate 0.9). Eval iter-4 champion on merged running.
- T6 cfg drafted: `sim_assemble_sarm_ext_iter5_2cam_train.json` (2-cam, merged, batch16, 5000 steps).