# SARM v4 + ACT/DP findings dump (pre-compact)

Date: 2026-05-04. Snapshot of long iteration: SARM v4 ceiling investigation, paper-recipe ablations, brainstorming, ACT/DP without RA-BC, CNN success classifier.

## TL;DR

- **SARM v4 best = `outputs/sim_3stage_sarm_v4_succ3w/checkpoints/007000` (mean_mid=0.302).** 30+ experiments tried. Plateau is **dataset issue**, not model.
- **v2 reproduction (sim_3stage_v2_no01_train_fs) reaches mean_mid=0.498-0.565** with same code → code is healthy, **v4 dataset is the bottleneck**.
- **Train/val are NOT held out at episode level** — `frame_stride_split` puts every 10th frame of EACH ep in val, the other 9/10 in train. The 0.302 ceiling is a TRAINING ceiling.
- **v3_eval dataset (`domrachev03/sim_3stage_v3_eval`, 15 held-out full eps)** is the proper eval set. baseline gets mean_mid=0.295 on it (basically same as in-distribution → not just an overfit-to-val issue, model is genuinely capped).
- **ACT v4_succ trained, doesn't succeed** — "stops halfway, grasps air" per user. iter1 (temporal ensembling) made it worse. iter2 (kl=1) and iter3 (chunk=8) training.
- **CNN binary success classifier WORKS** — overall=0.96, success_recall=1.0, nonsucc=0.96 with proper success definition.

## Datasets (HF cache `domrachev03/`)

| name | purpose | size | notes |
|---|---|---|---|
| `sim_3stage_v4_success_train_fs` | SARM training (success eps, frame-strided 90%) | 160 eps × 254 fr avg = 40640 | 6-stage anno, 7-D joint state, 128x128 2-cam |
| `sim_3stage_v4_val_fs` | SARM val (10% stride + 30 partials) | 190 eps × 6837 fr | TRAIN/VAL OVERLAP 160 eps |
| `sim_3stage_v4_with_partials` | full no-stride v4 (success+partials) | 190 eps × 69077 fr | needed for full-traj eval |
| `sim_3stage_v3_eval` | **HELD-OUT eval** | 15 eps × 3911 fr, 224x224 | RECORDED FRESH, primary SARM eval target |
| `sim_3stage_v4_3stage_*` | merged 3-stage variants | symlinks to v4 data | I built these |
| `sim_3stage_v4_eepose_*` | FK-derived EE pose proprio (8-D xyz+quat+grip) | symlinks to v4 data | I built these |
| `sim_3stage_v2_no01_train_fs` | v2 reproduction source | 139 eps × 51825 fr | gives 0.498-0.565 with same recipe |
| `sim_3stage_with_partials_v2` | v2 full-traj eval | 158 eps × 72735 fr | for v2_repro full-traj |

## Best ckpts

- `outputs/sim_3stage_sarm_v4_succ3w/checkpoints/007000` — SARM baseline best (mean_mid=0.302)
- `outputs/sim_3stage_sarm_v2_repro/checkpoints/014000` — v2 repro (mean_mid=0.565, full-traj 0.998 max)
- `outputs/cnn_success_cls/best.pt` — CNN success classifier (overall=0.96, success_recall=1.0)
- `outputs/act_v4_succ/checkpoints/last/pretrained_model` — ACT baseline (40k, behaviors visible but stops halfway)

## Experiments tried — SARM v4

All evaluated on val_fs (train/val overlap caveat) AND v3_eval (held-out 15 eps).

### Baseline + early-stage findings
- baseline succ3w 7k (wrist, sw=3, frame_gap=20): val mean_mid=0.302, v3_eval mean_mid=0.295 ★
- succ3 14k (2cam): 0.281
- 224x224: WORSE (0.104)
- partials in train: WORSE (0.029-0.059)
- 24k steps: WORSE (overfits)

### Paper-recipe ablations (all worse than baseline)
- M1 front cam: val=0.231/0.180
- M2 no causal mask: 0.208/0.204
- M3 MLP heads (`mlp_heads=true`): 0.176/0.157
- M4 ep_start anchor (`epstart_anchor=true` + `center_idx=1` fix): 0.188/0.207 (val), 0.144 v3_eval avg
- M5 sw=1 + 1500 steps: 0.210
- abl_sw1_short, abl_nocausal, abl_mlpheads — all losers

### Combos (paperfix = M2+M3 + nn.Linear stage prior + no-zero-on-lang-perturb)
- paperfix wrist: val=0.222/0.238, v3_eval=0.143 (worse on argmax_acc 0.295)
- paperfix_nostate (paperfix + disable_state): COLLAPSED (mean_mid=0.000 v3_eval avg)
- paperfix_short (3500 steps): pending eval

### 3-stage merge variants
- 3stage_2cam: val=0.149/0.142
- 3stage_wrist: val=0.121/0.122
- 3stage_2cam_invfreq (E2): val=0.232/0.215 (BEST 3-stage)
- 3stage_2cam_epstart_invfreq: val=0.147/0.220
- 6stage_2cam_invfreq: val=0.191/0.210 (regressed!)
- 6stage_wrist_invfreq: val=0.092/0.104 (regressed!)
- **invfreq is 3-stage-only; hurts 6-stage**

### EE pose / state experiments
- 6stage_wrist_eepose: val=0.100/0.129 (worse, joints > EE pose)
- 6stage_2cam_eepose: similar
- 6stage_2cam_invfreq_eepose: WORST so far (val=0.146/0.159)
- nostate variants (vision-only): all collapsed (mean_mid≈0)
- noise005/noise01 (state Gaussian noise σ=0.05/0.1): collapsed (0.075/0.054)
- short3500 (3.5k steps, early stop): val=0.242, v3_eval mean_mid=0.335 (best on mean_mid but max=0.636)

## Brainstorming Phase 1 — 4 parallel agents (all run on this codebase + paper)

### Agent A code audit findings (most actionable)
1. **Lang perturbation zeroed targets** (taught model "ignore vision when text wrong"). Paper doesn't. **FIXED** in `processor_sarm.py`.
2. **Stage prior was rank-≤6 in 768-d space** (zero-pad hack in `_stage_to_dmodel`). **FIXED** with `nn.Linear(max_stages, d_model)` (≡ nn.Embedding lookup).
3. Default `mlp_heads=False` deviates from paper.
4. Default `use_causal_mask=True` deviates from paper bidirectional.
5. Subtask token doesn't get positional bias (consistent but minor).
6. No gradient clipping anywhere in train step.
7. ReLU in fusion (paper unspecified, GELU more common).
8. Item 22: argmax of stage_pred has no straight-through; stage_model trained purely on its own CE — varying gt_stage_ratio doesn't help (verified).

### Agent B dataset audit findings
1. **CRITICAL — frame-stride is NOT episode-level holdout.** `frame_stride_split_3stage_v2.py:150`: `is_val = (frame_in_ep % 10) == 9`. Train and val share 160 episodes, 50ms apart per frame.
2. **84/800 stage transitions ambiguous** — `_adjust()` walks back through skipped frames; sometimes collapses `start[i+1]` onto `end[i]`. ~10% of transitions.
3. **Operator-press stage labels** = 100-300ms reaction lag = 2-6 frame noise per transition. No automatic verification.
4. **2 eps with 1-2 frame `bring_box`** (eps 38, 158) — degenerate labels.
5. **Last stage tail idle**: `place_cover_on_the_box` mean=93.5 frames, max=225. Operator presses success well after cap is on. Late-stage τ stretches over idle period.
6. **Episode lengths stereotyped**: v4 mean=254±41 (range 182-396) vs v2 mean=373±218 (171-720). 5x less variance.
7. **Stage durations stereotyped** (cv): bring_box v2=2.33 vs v4=0.24 (10x), approach_target v2=2.33 vs v4=0.19 (12x). Trivially predictable from time-since-reset.
8. **`object_spawn_offset=[0.1, 0, 0]` shifts every reset by same offset**. Per-object jitter only ±0.02 m + ±π/4 yaw.
9. **EE bounds tight**: z∈[0.33, 0.38] (5 cm), x∈[-0.14, 0.18], y∈[-0.77, -0.4845]. State[0..4] all have std<0.12 rad.
10. **Gripper saturates at ~0.79, never reaches 1.0** — essentially binary.
11. **Cap rgba=`0.2 0.2 0.2 1`** (dark) — combined with 128x128, cap may be near-invisible from front cam in approach phase.
12. **Tasks string says "Three-stage assembly"** for 6-stage dataset (cosmetic; CLIP text encoding sees wrong text).

**Agent B's recommendations for re-recording**:
A. Episode-level val split (16 held-out eps).
B. Fix `_adjust()` overlap.
C. Diversify object spawns + widen EE bounds + add deliberate pauses.
D. Auto-annotate stages from object pose (eliminates operator lag).
E. Drop tail idle.
F. 224×224 resolution.

### Agent C unexplored axes (most promising)
1. **Image augmentation** (color jitter, blur) — disabled CLIP cache + retrain
2. **CLIP patch tokens** (50/img) instead of CLS (~1)
3. **Bigger visual backbone** (DINOv2-B, ViT-L/14)
4. **Smaller transformer** (4 layers, hidden=512, 60M total to match paper)
5. **Validation-driven mid-train early stopping**
6. Label smoothing on stage CE
7. Focal loss
8. Dropout sweep (default 0.1, never moved)
9. Weight decay sweep (default 1e-3, never moved)
10. Gradient clipping
11. EMA on weights
12. AdamW betas (0.9, 0.95)

### Agent D paper deep-dive findings
1. Paper IS clip-vit-base-patch32 frozen (matches us).
2. Paper uses **9 frames = 1 ep_start anchor + 8 consecutive at 1s gap**.
3. Paper has **positional embedding ONLY on first frame** (matches our `first_pos`).
4. Paper has **2-layer MLP heads hidden=512** (we default to 1-layer).
5. Paper trains **2 epochs at batch=64** on single 4090. We train ~8.6 epochs at batch=32 → likely overfit.
6. Paper Table 6: **drop joint state → ρ 0.72→0.94**. They keep state as default but it hurts.
7. Paper has stage embedding fed into subtask transformer; our `_stage_to_dmodel` was zero-pad bottleneck (fixed via nn.Linear).
8. Subtask predicts within-stage τ ∈ [0,1], reconstructs y = P_{k-1} + α_k·τ. Our sigmoid output achieves this.
9. **Paper does NOT zero perturbed-language targets** (Agent A bug 1).
10. **Paper does NOT use causal mask** (bidirectional encoder).
11. Authors emphasize 60M param sweet spot. Ours is 120M (split into stage+subtask transformers, separate parameters).
12. Two-scheme training (sparse+dense union) significantly helps in paper Table 1.

## ACT v4_succ findings

### Trained
- `outputs/act_v4_succ` — chunk=20, n_obs=1, kl=10, ResNet18, 40k steps, batch=16
- behavior: approaches box, reaches halfway, grasps air. 0/10 sim success.

### iter1 (temporal ensembling τ=0.01, n_action_steps=1, eval-only — no retrain)
- 0/10 sim success, CNN max P(success) = 0.094 (vs baseline 0.249) — **worse**

### iter2 (kl_weight=1.0, retrain, 40k) — DONE 2026-05-04
- 0/10 sim, **avg_max_p=0.153 max=0.482** — worse than baseline. KL ablation rules out variance flattening.

### iter3 (chunk_size=8, n_action_steps=4, retrain, 40k) — DONE 2026-05-04
- 0/10 sim, **avg_max_p=0.025 max=0.070** — catastrophic. Smaller chunk hurt severely.

### iter4 (no-VAE, 40k) — DONE 2026-05-04
- 0/10 sim, **avg_max_p=0.066 max=0.300** — worse than baseline. CVAE prior was helping (counter to paper claim that CVAE only matters for sub-optimal demos).

### iter5 (80k steps, baseline cfg) — DONE 2026-05-04
- 0/10 sim, **avg_max_p=0.108 max=0.355** — worse than 40k baseline (0.249). Longer training hurt: overfit on stereotyped demos.

### **Final ACT board — all 5 variants worse than baseline**

| Run | sim | avg_max_p | max_max_p | n≥0.5 |
|-----|-----|-----------|-----------|-------|
| **baseline kl=10 chunk=20 40k** | 0/10 | **0.249** | 0.595 | **2** |
| iter1 temp_ens | 0/10 | 0.094 | 0.372 | 0 |
| iter2 kl=1 | 0/10 | 0.153 | 0.482 | 0 |
| iter3 chunk=8 | 0/10 | 0.025 | 0.070 | 0 |
| iter4 no-VAE | 0/10 | 0.066 | 0.300 | 0 |
| iter5 80k | 0/10 | 0.108 | 0.355 | 0 |

## DP v4_succ — DONE 2026-05-04
- 50k steps, batch=64, n_obs=2, horizon=16, n_action=8, ResNet18 with `use_group_norm=false`
- 0/10 sim, **avg_max_p=0.018 max=0.033** — catastrophic. DP fails harder than ACT on this dataset.

## CNN success classifier

### Final recipe (works)
```
python scripts_local/train_cnn_success_classifier.py \
  --epochs 8 --samples-per-epoch 1024 --batch 64 \
  --pos-weight 10.0 --lr 2e-4 --freeze-stages 0 \
  --image-key observation.images.front --ratio 1
```
- ResNet18 unfreezed, ImageNet pretrained
- 1:1 sampling (success:non-success — necessary because raw imbalance is too large)
- pos_weight=10 in CE loss
- success = last frame of each ep (where `next.reward >= 0.5`)
- non-success: 50% from `place_cover_on_the_box` BEFORE success start, 50% from earlier stages
- Drop eps with no success_start>0 or no stage5

### Eval on v3_eval (15 held-out demos)
- overall=0.960, success_recall=1.000 (15/15), nonsucc=0.960
- ckpt: `outputs/cnn_success_cls/best.pt`

### CNN on rollouts (act_v4_succ)
- baseline: 2/10 eps reach P≥0.5 briefly, 0/10 reach 0.8, avg max=0.249
- iter1 (temp_ens): 0/10 reach P≥0.5, avg max=0.094
- Saved at `/tmp/cnn_rollout_predictions/`

## Code changes (uncommitted, lerobot_policy_sarm)

### configuration_sarm.py — added cfg knobs
```
use_causal_mask: bool = True       # M2: paper bidirectional
mlp_heads: bool = False             # M3: paper 2-layer MLP head hidden=512
disable_state: bool = False         # zero proprio (paper Table 6 says drop)
epstart_anchor: bool = False        # M4: token[0] = ep_start, paper-style
state_noise_std: float = 0.0
time_warp_prob: float = 0.0
```

### modeling_sarm.py
- `StageTransformer.__init__` + `SubtaskTransformer.__init__` accept `use_causal_mask`, `mlp_heads`, `max_stages`.
- `_stage_to_dmodel` now uses learned `nn.Linear(max_stages, d_model)` instead of zero-pad.
- Conditional causal mask in `forward` of both transformers.

### processor_sarm.py
- Don't zero (stage, τ) targets when language is perturbed.
- Optional Gaussian noise on state during training (`state_noise_std`).
- `disable_state` zeroes state after pad.

### sarm_utils.py
- `compute_absolute_indices` accepts `epstart_anchor=True` → `[-1e6, 0, gap, 2*gap, ..., 7*gap]`.

### sarm.py + eval_sarm_sim_assemble.py
- `center_idx = 1` when `epstart_anchor=True` (else `n_obs//2`).

## Beads

- **lerobot-93** EPIC: SARM v4 paper-recipe ablation (in_progress)
- **lerobot-115**: brainstorm Phase 1 done — fixes in paperfix combo, didn't help
- **lerobot-117** EPIC: ACT v4 'stops halfway' iteration — **all 5 iters complete, all worse than baseline**
  - lerobot-118 iter1 (temp_ens) — failed (0.094)
  - lerobot-119 iter2 (kl=1) — failed (0.153)
  - lerobot-120 iter3 (chunk=8) — failed (0.025)
  - lerobot-121 iter4 (no-VAE) — failed (0.066)
  - lerobot-122 iter5 (80k) — failed (0.108)

## Memory updates this session

- `feedback_no_kill_in_progress.md` — never pkill running training without explicit ask
- `feedback_watch_eval_completion.md` — arm watcher on every long-running eval
- `feedback_eval_on_same_gpu.md` — pin CUDA_VISIBLE_DEVICES of eval to GPU that just freed
- `reference_v3_eval_dataset.md` — `domrachev03/sim_3stage_v3_eval` (15 held-out, primary SARM eval)

## Next-step priorities (post 6/6 model-side failures)

Model-side hyperparameter sweep is **exhausted**: 5 ACT variants + DP all worse than the baseline ACT run that already gets 0/10 sim success. Combined with the SARM v4 ceiling (0.302 vs v2_repro 0.498-0.565 with same code), the bottleneck is unambiguously the v4 dataset.

Dataset-side actions to consider (ranked by effort):
1. **Re-record demos** with cleaner stage transitions (reduce the 84/800 ambiguous transitions) and looser `_adjust()` walk-back. v4 transition stereotypy (cv 7-12x lower than v2 in middle stages) is suspicious — too much determinism.
2. **Operator-press lag fix**: shift action labels backward by ~100-300ms to compensate for human reaction delay between visual cue and gripper actuation.
3. **Augmentation in training**: stronger image transforms (already enabled), Mixup of demos, action noise.
4. **Drop tightest stages from supervision**: focus ACT on pre-grasp + post-grasp segments only.
5. **Bigger backbone**: ResNet50 or DINOv2 features (untested).
6. **CLIP patch-token visual tokens** (untested in SARM/ACT).

For SARM: stop pursuing model-side fixes. Either accept 0.302 baseline or re-record before more SARM work.

## Key infra notes

- DL_A6000: 4 GPUs, ~30-50GB free disk (tight). Always rsync code (no git). Use `~/.local/bin/uv` (not in PATH).
- Local PC: 1 GPU, 16GB. Train ACT/DP fast (37 step/s for ACT, 10 step/s for DP).
- Sim mujoco rendering needs `MUJOCO_GL=egl` for headless eval.
- ACT eval with `eval_chunk_policy` writes side-by-side front+wrist mp4s to `--video-dir`.
- ACT cfg requires `n_obs_steps=1` (multi-obs not supported).
- DP cfg needs `use_group_norm: false` when using ImageNet-pretrained backbone.
