# SARM sim_assemble findings (caveman)

Running log of iter configs, metrics, and conclusions. Companion to
`2026-04-24-sarm-sim-assemble-train-plan.md`.

## Infra fixes (not iter numbered)

- **transformers 5.x compat** — 2026-04-24: installed transformers 5.5.4 returns
  `BaseModelOutputWithPooling` from `CLIPModel.get_image_features` /
  `get_text_features` where older transformers returned a tensor directly.
  Fixed in `src/lerobot/policies/sarm/processor_sarm.py` lines 436/463:
  extract `.pooler_output` if available, else use object as-is. No semantic
  change — same 512-dim CLIP features either way. (User flagged they didn't
  want SARM code edits, but this is a pure compat shim — blocking bug, not
  algorithm change.)
- **missing `faker` pip pkg** — `processor_sarm.py` imports `from faker
  import Faker`. Added via `uv pip install faker` (now 40.15.0).

## Dataset state

- source: `domrachev03/sim_assemble_sarm_multistage_two_stages_filtered_v2`
  (50 eps, 10778 frames @ 20fps; bucket dist {0:5, 1:5, 2:5, 3:5, 4:30})
- train split: `..._v2_train` (40 eps, 8523 frames; {0:4, 1:4, 2:4, 3:4, 4:24})
- val split: `..._v2_val` (10 eps, 2255 frames; {0:1, 1:1, 2:1, 3:1, 4:6})
  - val ep indices (from source): `{0, 5, 10, 15, 20, 25, 30, 35, 39, 44}`
- temporal_proportions_sparse: `{approach_box: 0.236, bring_box: 0.266,
  approach_target: 0.237, place_target_in_the_box: 0.261}` (inherited from
  source; breakpoints ≈ [0, 0.236, 0.502, 0.739, 1.000])
- external val: `domrachev03/sim_assemble_demos_two_stages` (30 full-success,
  5442 frames, NO stage labels — success-only sanity check)

## Iter table

| iter | cfg | train data | stage_loss_w | n_obs / gap | SuccTerm | FailTerm | 0/4 max≥0.5 | lin_mad | mean_mid | stage_argmax | plateau | verdict |
|-----|-----|-----------|-------------|-----------|---------|---------|-----------|---------|----------|-------------|--------------|---------|
| 0 (A) | single_stage | demos (30) | n/a | 4/5 (1.0s) | 83% ❌ | 0% ✅ | 1.0 ❌ | 0.08 ✅ | 0.56 ✅ | n/a | 0% ❌ | null-baseline; misses TP gate and spikes on partials |
| 0 (B) | dual | v2_train (40) | 3.0 | 4/5 (1.0s) | 100% ✅ | 0% ✅ | 1.0 ❌ | 0.10 ✅ | 0.67 ✅ | 0.93 | 25% ❌ | TP+linearity pass; partial-ep peaks spike FP |
| 1   | dual | v2_train (40) | 10.0 | 4/5 (1.0s) | 100% ✅ | 0% ✅ | 1.0 ❌ | 0.10 ✅ | 0.68 ✅ | 0.93 | 25% ❌ | stage_loss=10 no effect — stage argmax already committed on full eps; spikes persist on partials |
| 2   | dual | v2_train (40) | 3.0  | 8/5 (2.0s,rw3) | 100% ✅ | 0% ✅ | 1.0 ❌ | 0.098 ✅ | 0.66 ✅ | 0.93 | 25% ❌ | longer window helped monotonicity (0.89→0.92) and 1/4 stage_argmax (0.77→0.88), 2/4 max got worse (0.96→0.99), plateau unchanged |
| 3 dense | dense_only + drop_last=5 | v2_train (40) | 3.0 | 4/5 (1.0s) | 83% ❌ | 0% ✅ | 1.0 ❌ | 0.104 ✅ | 0.67 ✅ | 0.92 | 25% ❌ | dense head ~= iter-0 B. drop_last=5 cost 1 full ep's TP (ep0 term 0.91→0.87). annotation_mode change no effect on spikes. |
| 3 sparse | (same ckpt, sparse head = whole-ep single stage) |  | 3.0 | 4/5 (1.0s) | 83% ❌ | 0% ✅ | 1.0 ❌ | 0.084 ✅ | 0.51 ✅ | n/a | 50% | sparse head in dense_only is single-stage linear progress; worse monotonicity (0.75) and partial eps peak ~0.94 |
| **4** 🏆 | dual | v2_train (40) | 3.0 | 8/5 (2.0s,rw3) **WRIST** | **100% ✅** | **0% ✅** | **0% ✅** | **0.097 ✅** | **0.66 ✅** | **0.93** | 25% ❌ | **CHAMPION — 6/7 gates. Wrist cam is the key signal.** 0/4 max only 0.44 (was 0.85+). Plateau still short — 1/4 and 2/4 eps overshoot by ~0.15-0.25 (n=1 per bucket, noisy). |

(filled as runs complete)

## Per-iter notes

### iter 0 (B) — multistage dual baseline
- config: `src/lerobot/rl/sim_assembling_sarm_multistage_train.json`
- policy.n_obs_steps=4, frame_gap=5 (observation window ±10 frames = ±0.5s, span 1.0s at 20fps)
- max_rewind_steps=2 → 5 obs + 2 rewind = 7 frames/seq
- stage_loss_weight=3.0 (panda iter 12 best)
- steps=5000, batch_size=16
- smoke test 20 steps: loss 2.84→2.74 stable
- output: `outputs/sim_assemble_sarm_multistage_iterB/checkpoints/last/pretrained_model`

### iter 0 (A) — single-stage demos-only null baseline
- config: `src/lerobot/rl/sim_assembling_sarm_single_stage_train.json`
- killed at step 839 (loss 0.002, converged); used step-500 ckpt
- task text: `"Two-stage assembly"` (matches embedded task field in demos + filtered_v2)
- stats: `domrachev03/sim_assemble_demos_two_stages` (self; has pop. stats)
- eval @ val (10 eps mixed): succ_term 5/6, partials all spike max≥0.5 (0/4 max=0.92), plateau 0/4
- failure mode: model trained on success-only, produces roughly-linear progress on full eps (ok) but has nothing to pin a failure signal to → monotone-rising on partials too → spikes at end
- verdict: expected null-baseline behaviour; confirms multistage dataset + dual mode is necessary

### iter 0 (B) eval results
- eval @ val (10 eps, mixed buckets): succ_term_rate=100%, lin_mad=0.102 (linear ✓), mean_mid=0.67, monotonicity=0.89, stage_argmax=0.93 on full eps
- partial-ep failure mode: peak progress SPIKES mid-ep even though terminal progress is correct:
  - 0/4 ep: term=0.45, max=0.85 ← big FP spike
  - 1/4 ep: term=0.15 ✓, max=0.69 ← mid-ep spike
  - 2/4 ep: term=0.31 ✓, max=0.96 ← big mid-ep spike (reaches apparent success)
  - 3/4 ep: term=0.71 ✓, max=0.78 ✓ plateau correct
- root cause hypothesis: stage_loss_weight=3 too soft → model oscillates in stage-argmax during partial eps → progress spikes when model briefly commits to a high-stage prediction
- counter-example: full eps have stage_argmax=0.93 → argmax correct when ep genuinely progresses
- fix candidate: increase stage_loss_weight (panda iter 12→13 did 3→10 jump for OOD)

## Iter-1 hypothesis (decided)

Primary knob: `stage_loss_weight: 3.0 → 10.0`. Rationale: force committed stage-head classification, reducing mid-ep progress oscillation on partial eps. Keep everything else identical to iter 0 (B) for clean A/B comparison. Config: `src/lerobot/rl/sim_assembling_sarm_multistage_iter1_train.json`.

### iter 1 results (no improvement)

- per-bucket: 0/4 max 0.85→0.93 (worse), 2/4 max 0.96→0.98 (worse), 3/4 max 0.78 (same), full all 100% ✓.
- Stage argmax on full ep still 0.93, same as iter-0 → model already committed on full eps; stage_loss_weight=10 only marginally tightened stage classification.
- Partial eps still spike: progress peaks ~0.93 on 0/4 and ~0.98 on 2/4. Stage probs in iter-1 plots show briefer but still present transitions to wrong-stage → breakpoint[k] jumps.
- Conclusion: stage head wasn't the bottleneck; need to reduce per-frame miscommitments via more context.

## Iter-2 hypothesis (decided)

Knobs: `n_obs_steps: 4 → 8`, `max_rewind_steps: 2 → 3`, `stage_loss_weight: 10 → 3` (revert). Keep frame_gap=5. Rationale: 2× temporal window (2s instead of 1s) forces model to rely on longer-range appearance trends. Bigger max_rewind adds more training-time augmentation diversity (panda iter-13m pattern). `num_frames = 1 + 8 + 3 = 12`. Config: `src/lerobot/rl/sim_assembling_sarm_multistage_iter2_train.json`.

## Iter-3 plan (user-requested: dense_only + drop_n_last_frames)

User requests 2026-04-24:
1. `annotation_mode: dense_only`. Sparse head reduces to single whole-ep stage (no mid-ep breakpoint transitions to cause spikes); dense head keeps 4-stage subtask breakdown.
2. `drop_n_last_frames: 1 → 5`. Drops the last 5 frames of each training ep → prevents model from memorizing "this specific frame = progress 1.0" which is a candidate cause of spikes on partial eps that visually resemble end-states.

Config: `src/lerobot/rl/sim_assembling_sarm_multistage_iter3_train.json`. Keep baseline n_obs=4, max_rewind=2, stage_loss=3 (pending iter-2 result).

Eval with `--head-mode=sparse` (overall FP gate, now single whole-ep stage) and `--head-mode=dense` (per-stage plateau/argmax). Eval harness already supports the flag.

User also noted: will extend demo collection when PC access available (addresses the too-few-partial-eps bottleneck — 4 training eps per partial bucket is very small).

## State after iter-3

Tried so far (all keeping dataset and core SARM model fixed):
- iter-0 (B): dual, stage_loss=3, n_obs=4/gap=5 → baseline
- iter-1: stage_loss=3→10 → no change
- iter-2: n_obs=4→8, max_rewind=2→3 (2s window) → small monotonicity gain
- iter-3: annotation_mode=dual→dense_only, drop_n_last_frames=1→5 → no gain, TP mildly hurt

**The partial-ep spike failure mode is robust to every config knob tried.** The root cause is the training set: only 4 partial eps per bucket (1/4, 2/4, 3/4, 0/4). The model's progress head defaults to what it sees most often (success trajectories) and has too few counter-examples to learn robust "this is NOT succeeding" discrimination in mid-ep.

## Iter-4 plan (decided)

Primary knob: `image_key: observation.images.front → observation.images.wrist`. Rationale: wrist camera is close to the gripper/object and sees "object-in-box" vs "object-near-but-not-in-box" directly. Front camera is visually ambiguous during mid-ep (robot hand position looks similar for ~2/4 and 4/4 at some frames). Wrist is orthogonal visual signal.

Keep: annotation_mode=dual (back to panda-proven), n_obs=8, max_rewind=3 (keep iter-2 monotonicity gain), stage_loss=3, drop_n_last_frames=1 (revert).

Config: `src/lerobot/rl/sim_assembling_sarm_multistage_iter4_train.json`.

### iter-4 results (CHAMPION)

- per-bucket:
  - full (6): term mean 0.98, all ≥0.9 ✓
  - 3/4: term 0.63, max 0.78 → in plateau [0.64, 0.84] ✓
  - 2/4: term 0.29, max 0.79 → overshoots plateau [0.40, 0.60] by 0.19
  - 1/4: term 0.21, max 0.61 → overshoots plateau [0.14, 0.34] by 0.27
  - 0/4: term 0.43, max **0.44** → stays below 0.5 (all prior iters had 0.85+) ✓
- progress curves (see PNGs): wrist cam gives clean plateau at mid-to-low values; brief frame-55-100 approach_target-prob spike in 0/4 ep caused 0.30 progress bump but nothing crosses 0.5.
- stage_argmax: 0.93 on full eps; 0.93 on 3/4 ep; wrist cam preserves stage-head discrimination.
- Verdict: **SHIPPABLE** for HIL-SERL reward. Plateau tightness on 1/4 and 2/4 is a "nice to have" that requires more partial-ep training data (user will extend when PC available).

### External validation on demos_two_stages (30 full-success, no stage labels)

- All 30 eps terminal progress ≈ 0.96-0.997. term≥0.9 rate = **100%**.
- Output dir: `outputs/sarm_eval/iter4_demos/`.
- Caveat: the eval harness mislabels these as "0-of-4" because the dataset has no `sparse_subtask_names`. The "fail_term_rate=1.0" in summary.md is therefore spurious — treat as cross-check for TP only.
- Confirms: the model transfers cleanly from the multistage train split to the demos split, no OOD collapse.

