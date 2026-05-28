# 2-stage subset diagnostic — does ACT learn HALF the 6-stage task?

**Beads epic:** lerobot-154

## Goal

Determine whether 0-15% ACT CNN succ on 6-stage v2 is task complexity (too many stages for 205 demos) or pipeline bug. Truncate demos at end of stage 4 → train ACT BC + RA-BC → if halfway succ ≥ 50%, complexity bottleneck confirmed.

## Stage layout (sim_3stage_v2)

```
idx  name                      proportion  cum
0    approach_box              0.124       0.124
1    bring_box                 0.130       0.254
2    approach_target           0.113       0.367
3    place_target_in_the_box   0.175       0.542  ← truncate HERE
4    approach_cover            0.148       0.690
5    place_cover_on_the_box    0.311       1.000
```

Truncation = keep frames `[0, sparse_subtask_end_frames[3] + 1)` per episode.

## Halfway CNN classifier spec

- **Positives** (label=1): last 5 frames of stage 4 per demo, i.e. frames `[end[3]-4, end[3]+1)`.
- **Negatives** (label=0):
  - 50% from frames `< sparse_subtask_start_frames[3]` (pre-place_target, scene looks "early")
  - 50% from frames in stages 5-6 in *full untruncated* ds (post-place, distractor: arm moves to cover)
- Image key: `observation.images.front`.
- Recipe: ResNet18 unfreezed, ImageNet pretrained, 8 epochs, batch 64, pos_weight 10, lr 2e-4, ratio 1.
- Output: `outputs/cnn_halfway_v1/best.pt`.
- Target: `success_recall ≥ 0.9` on held-out demos' stage-4-end frames.

## Eval success criterion (per-episode)

Both reported, "halfway success" = AND of:
1. `max_SARM_cum_progress >= 0.50` (K SARM, sync_inference=true)
2. `max_halfway_CNN_prob >= 0.5` within episode

Per-variant aggregate: `halfway_succ_rate = #eps satisfying both / n_eps`.

## Training recipe (v11 from lerobot-52 win)

```
chunk=10, n_action_steps=10, n_obs_steps=1
VAE on, kl=10, ResNet18 ImageNet
MIN_MAX state+action, MEAN_STD visual
lr=1e-5, batch=16, 80k steps
```

Variants:
- **BC**: pure BC, no RA-BC.
- **RA-BC κ=0.30**: same + `use_rabc=true, rabc_kappa=0.30, rabc_progress_path=<from T5>`.

## Eval env

Clone `sim_3stage_act_eval_env_K_sync_fix4_img128.json`. Lower `terminate_on_success.threshold` 0.99 → 0.50. Keep image_size=[128,128], EE bounds, K SARM sync.

## Decision matrix (T12)

| BC succ | RA-BC succ | Conclusion |
|--------|-----------|------------|
| ≥50%   | ≥50%      | Complexity bottleneck. Action: re-record more demos OR Pi0 finetune. |
| <30%   | ≥50%      | RA-BC critical. Action: scale RA-BC + more demos on full 6-stage. |
| ≥50%   | <30%      | RA-BC overfilters truncated data. Investigate weight distribution. |
| <30%   | <30%      | Pipeline bug. Escalate — re-examine action repr, state, image normalization. |
| 30-50% range | — | Marginal; iterate κ. |

## RESULTS (2026-05-12)

### Training (DL_A6000, ~70min each)

| variant | output_dir | final loss | grdn | step/s | ckpt |
|---|---|---|---|---|---|
| BC | act_v2_first4_bc_v11 | 0.065 | 1.92 | 14-21 | 80000 |
| RA-BC κ=0.30 | act_v2_first4_rabc_K_kappa30_v11 | 0.067 | 2.19 | 14-21 | 80000 |

### Eval w/ SARM termination (threshold 0.50, 10 eps each, local G0)

| variant | SARM succ_rate | halfway_CNN succ_rate | ep_lens | cnn_max range |
|---|---|---|---|---|
| BC | 10/10 (1.0) | 0/10 (0.0) | 21-56 | 0.001-0.015 |
| RA-BC κ=0.30 | 10/10 (1.0) | 0/10 (0.0) | 21-56 | 0.001-0.010 |

**SARM K hallucinates** — claims 60-88% completion in 21-56 steps (1-3s) when actual demo halfway takes 134+ frames.

### Eval w/o SARM termination (30s full eps, 5 eps each)

| variant | cnn_succ_rate | cnn_max max | max_cum max |
|---|---|---|---|
| BC noterm | 0/5 | 0.014 | 0.882 |
| RA-BC noterm | 0/5 | 0.017 | 0.888 |

### CNN ground-truth verification

Halfway CNN (`outputs/cnn_halfway_v1/best.pt`, overall 0.958 acc, pos 0.922 on held-out demos):

- **Demo ep0 last 10 frames** (true halfway success state): P(succ)=1.000 unanimously. CNN works.
- **BC rollout ep01 (SARM=0.88)**: P(succ)=0.001-0.004 across all 600 frames. Robot visually NEVER approaches halfway state.

### Diagnosis

**Truncation diagnostic NEGATIVE.** ACT on 4-stage task fails identically to full 6-stage:
- Both BC and RA-BC: 0% true halfway succ over 30s eval.
- Halfway CNN correctly fires on demo end-frames (1.0) but rejects all rollout frames (<0.02).
- SARM K hallucinates "halfway done" within 1 sec on policies that did not actually move toward target.

**Bottleneck is NOT task complexity.** Halving the task did not help. Same 0% CNN-verified result.

### Eliminated hypotheses
- ✅ Task complexity (6-stage too long): refuted — 4-stage also 0%.
- ✅ Demo count (205 too few for 6-stage): refuted — same demo set, half the stages, still 0%.
- ✅ RA-BC weight quality: refuted — RA-BC matches BC, neither moves to halfway.

### Remaining hypothesis: **pipeline-level bug**

What's still in play (none ruled out by this diag):
1. Action representation mismatch demo↔eval (already verified action scale, but maybe action TIMING/ordering).
2. Image preprocessing mismatch (resize/crop/color order between training input and eval observation).
3. State preprocessing (normalization range).
4. CVAE/encoder pathology — ACT learns deterministic open-loop pattern (drifts +X) regardless of obs.
5. Demo→env distribution shift in some axis we haven't measured (initial object pose distribution, lighting, etc.)

### Recommended next steps (in order)

1. **Demo replay w/ halfway CNN.** Run `replay_demo_in_env` w/ CNN classifier on actual demo actions in eval env. If demo actions also fail halfway CNN → env-distribution issue. If they pass → ACT-specific failure.
2. **Inspect ACT rollout videos vs demo videos side-by-side.** What's visually different about ACT's trajectory.
3. **Sanity check ACT outputs on training set.** Sample observed state, feed through trained ACT, compare predicted action to recorded action. If they match → ACT learned demos correctly. If diverge → training broken.
4. **Test ACT with chunk_size=80 on truncated ds.** Previous full-6-stage best was κ=0.30 chunk=80. Maybe v11 chunk=10 recipe is suboptimal for 6-stage even truncated.
