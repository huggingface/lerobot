# SARM 3-stage iter6+ plan (extended partials, balancing)

epic lerobot-60. start 2026-04-30 evening. previous champion = iter3-14k stuck at stage_not_below=0.55.

## what changed

user extended `domrachev03/sim_assemble_sarm_multistage_three_stages_failures` from 19 → **58 eps** with full plateau coverage:

| n_stages | count | notes |
|---|---|---|
| 0/6 | 9  | env reset / no movement / immediate fail |
| 1/6 | 10 | stuck after approach_box |
| 2/6 | 10 | stuck after bring_box |
| 3/6 | 10 | stuck after approach_target |
| 4/6 | 9  | stuck after place_target_in_the_box |
| 5/6 | 10 | stuck after approach_cover |

(user said 40 added but only 39 actually present — 0/6 has 9 not 10. negligible.)

prior partials iter (iter3) had ONLY 4/6 + 5/6. now we get all early-stage plateaus.

## eval scope change

per user: **drop old 2-stage datasets from eval**. eval only on new val split. so:
- eval target: `domrachev03/sim_3stage_partials_val_fs_v2` (frame-stride split of merged 158 ep ds)

## class imbalance estimate (frame-level)

with full 800-frame timeouts on partial eps, frame distribution is heavily skewed:
- 0/6 ep contributes 800 frames, all sitting in stage 1 (approach_box) → stage 1 will dominate
- 1/6 ep: ~most frames in stage 1, few in transition out
- success ep (~263 frames avg from 100 succ ds): more even, ~40 frames per stage average

rough back-of-envelope frame counts per stage (sparse labels):

| stage | succ frames (100×~263) | partial frames (58×800 distributed) | partial:succ ratio |
|---|---|---|---|
| approach_box | ~4400 | ~30000 | 6.8× |
| bring_box | ~4400 | ~25000 | 5.6× |
| approach_target | ~4400 | ~18000 | 4.0× |
| place_target | ~4400 | ~10000 | 2.3× |
| approach_cover | ~4400 | ~5500 | 1.2× |
| place_cover | ~4400 | ~0 | 0× |

→ **partial-fail injection skews stage_loss heavily toward early stages**. terminal `place_cover` gets zero negative examples. likely root cause of low term-rate.

## plan

### T8 (in progress): data prep
1. outlier scan partials ds (drop NaN / frozen / duplicates)
2. drop confirmed bad eps
3. merge `sim_assemble_sarm_multistage_three_stages_success` (100) + filtered partials → `domrachev03/sim_3stage_with_partials_v2`
4. frame-stride split (N=10) → `_v2_train_fs` + `_v2_val_fs`
5. build CLIP cache for both splits
6. write temporal_proportions json

### T9: iter6 baseline train
- cfg: copy iter3 cfg, swap dataset → `_v2_train_fs`
- arch: sarm_ext, n_obs=8, gap=5, batch=32, stage_loss_weight=3, 14k steps
- output: `outputs/sim_3stage_sarm_iter6/`

### T10: iter6 eval (new val only)
- eval on `_v2_val_fs` only
- gates: succ_term≥0.95, stage_ne≥0.90 (#1), lin_mad≤0.25 (#2), stage_nb≥0.70 (#3), monotonicity≥0.85, plateau_ok≥0.80
- bucket breakdown: full (100) + 0/6..5/6 partials (58)

### T11: stage-balance brainstorm

if T10 fails, brainstorm balancing.

#### candidate techniques (ranked by expected leverage)

| # | technique | mechanism | expected leverage | effort |
|---|---|---|---|---|
| 1 | **per-stage class weight in stage CE loss** | weight = 1/freq, applied in modeling_sarm.py stage_head loss | hi — directly addresses imbalance at loss level | low (1 line + cfg arg) |
| 2 | **balanced batch sampler** at frame level | each batch contains roughly equal frames per stage class | hi — every batch teaches all stages | mid (custom torch sampler) |
| 3 | **per-bucket episode sampler** | each batch has equal eps from {full, 0/6, 1/6, 2/6, 3/6, 4/6, 5/6} | mid — coarser than #2, but matches episode-level imbalance | mid |
| 4 | **undersample early-plateau partials** to cap each stage at ~target frame count | brute-force balance | mid | low (preprocessing) |
| 5 | **focal stage loss** (γ=2) | downweight easy/well-predicted stages | mid — adaptive, no manual weights needed | low |
| 6 | **terminal-frame oversampling** | upsample frames near `place_cover` for succ eps | hi — directly boosts term-rate signal | low (sampler config) |
| 7 | **truncate partial eps to last N frames before plateau** | drop most pre-plateau frames so partials only contribute "stuck near boundary" frames | mid-hi | mid |

#### picked combo (T12 iter7)
**1 + 6**: class-weighted stage loss + terminal-frame oversampling.
- rationale: lowest effort, addresses BOTH disbalance dimensions (stage loss weighting AND term-rate signal). #2 (balanced sampler) is a second-line fallback if #1+#6 don't suffice.

#### later iters if needed
- iter8: add #2 balanced sampler
- iter9: try #5 focal loss
- iter10: combine #1+#7 (truncate partial tails)

### T12+: iterate till pass

stop conditions:
- succ_term ≥ 0.95 on full bucket of new val (primary gate)
- AND stage_not_exceed ≥ 0.90 on all buckets (priority #1)
- AND stage_not_below ≥ 0.70 on all buckets (priority #3)
- AND lin_mad ≤ 0.25 (priority #2)
- AND plateau_ok ≥ 0.80 on partial buckets

## risks / known traps

- **`_3` ds (memory)**: not used here. just current 58 partials are fresh.
- **iter6 2-stage precedent**: mixed-source partials caused regressions on full-success eps. mitigate by checking stage_ne on full bucket every iter.
- **stage 6 zero coverage**: NO partials end at stage 5 + plateau into stage 6 with failure. all our 5/6 eps stop AT `approach_cover`, not after attempting `place_cover_on_the_box`. so the model never sees "tried place_cover, failed" — only "never tried". this might cap term-rate even with balancing.
- **0/6 = scene reset failure?**: 9 eps with `names=None` and length=800 are odd (no even approach_box reached). need to spot-check whether these are real failures or data corruption.

## bug discovery during iter6 prep

`stage_loss_weight` field exists in `SARMConfig` but was **NEVER applied** in `modeling_sarm.py`.
`total_loss = stage_loss + subtask_loss` (the weight was dead code).

⇒ **iter3/iter4/iter5 differences in `stage_loss_weight` (3.0/6.0/3.0) were no-ops**. All three trained
with effective weight=1.0. Their performance differences were entirely from other knobs (steps, n_obs).

Fixed in iter7 codepath: `total_loss = config.stage_loss_weight * stage_loss + subtask_loss`.

## actual stage-frame balance (post-merge, train)

| stage | frames | % |
|---|---|---|
| approach_box | 10691 | 16.3% |
| bring_box | 10273 | 15.7% |
| approach_target | 9574 | 14.6% |
| place_target_in_the_box | 11014 | 16.8% |
| approach_cover | 9042 | 13.8% |
| place_cover_on_the_box | 8482 | 12.9% |

surprisingly balanced (12.9%-16.8%). reason: partials' stuck tails go to the LAST listed stage, which
scatters load across box/bring/target/place_target/cover. only place_cover_on_the_box is success-only.

**implication: stage_class_weights="inverse_freq" is a mild correction**. The "imbalance" the user
flagged is at the EPISODE-bucket level (100 succ vs ~10 per partial bucket), not stage-frame level.

## refined iter7+ plan

- **iter7 (T12)**: stage_loss_weight=1.0 + stage_class_weights="inverse_freq" + apply stage_loss_weight bugfix.
  Mild reweighting test. If marginally helpful, escalate.
- **iter8 if iter7 marginal**: stage_loss_weight=3.0 (emphasize stage CE) + same class weights.
  This is what users THOUGHT iter3-5 was doing but wasn't.
- **iter9 if still failing**: per-episode-bucket weighted sampler (each batch ≈ equal eps per bucket).
  Implementation: custom torch Sampler returning episode indices balanced across {0/6, 1/6, 2/6, 3/6, 4/6, 5/6, 6/6}.
- **iter10**: tau-loss reweighting — boost MSE for terminal frames (last 10% of success eps).
  Targets the succ_term gate directly.

## results

(populate as iters complete)
