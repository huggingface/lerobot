# SARM/ACT v3 next-steps brainstorm

session: 2026-04-30 → 2026-05-01. epic lerobot-80 (active).

## state recap

| component | state |
|---|---|
| env | UPDATED: 224×224 cams, cam_front (0,-0.85,0.35), dark cap rgba=0.2, ee_bounds.max[0]=0.18, min[0]=-0.14, min[1]=-0.77 |
| ds | RECOLLECTED: sim_3stage_v3_success (60 eps), sim_3stage_v3_failures (30 eps after dropping accidental ep21) |
| merged train ds | sim_3stage_v3_train_fs (89 eps, 36737 frames, 90% frame-stride split) |
| merged val ds | sim_3stage_v3_val_fs (89 eps, 4131 frames, 10% frame-stride split) |
| CLIP cache | built on both splits |
| prod SARM (legacy) | sim_3stage_sarm_iter8 (works on v3 data: max=0.94 on full success) |
| prod ACT (legacy) | act_3stage_v3 (chunk=20, 80k, mean max=0.76 at thr=0.95 — 0/20) |
| v3-iter1 stock train | 87% / 14k, loss=0.001 (clean convergence), eval pending |

## SARM v3 next iters (decision tree)

### branch A: v3-iter1 passes all gates
- ship as `outputs/sim_3stage_sarm_v3_iter1` production
- update teleop env cfg + ACT eval cfg to point at v3-iter1
- pivot to ACT v3 retrain on new ds

### branch B: v3-iter1 partial pass (likely outcome by past pattern)
gates likely failing: stage_not_below, plateau_ok, monotonicity. Apply v2 lessons:

| iter | lever | rationale |
|---|---|---|
| **v3-iter2** | drop 0/6 partials (`no0_train_fs`) | replicate v2 iter7 win on plateau gates (0/6 visually ambiguous with success-start) |
| v3-iter3 | drop 0+1/6 partials | replicate v2 iter8 (best succ_term per gate priority) |
| v3-iter4 | retrain v3-iter1 cfg + apply stage_loss_weight=3 (bug fix) | replicate v2 iter9 bug-fix experiment |

### branch C: v3-iter1 collapses again (max < 0.5)
- Diagnostic: instrument target generation, check actual GT distribution
- Try increasing `n_obs_steps` to 12 (longer context window on the new 224 ds)
- Try `frame_gap=10` (was 5; longer temporal sweep)

## architectural / hyper-tuning levers (cross-branch)

| | lever | leverage | risk |
|---|---|---|---|
| 1 | n_obs_steps 8→12 | mid (longer ctx for 6-stage horizon) | minor — uses more state |
| 2 | frame_gap 5→10 | mid (broader temporal view) | mid — too coarse may miss transitions |
| 3 | stage_class_weights="inverse_freq" + bugfix applied | mid (auto-balance stage CE) | mid — may underweight terminal stage |
| 4 | longer training 14k→24k | lo (loss already 0.001, plateau likely reached) | none |
| 5 | annotation_mode "dense_only" instead of "dual" | lo | risky — dual gave best v2 result |
| 6 | hidden_dim 768→1024 (bigger transformer) | lo (compute-bound, not capacity-bound) | hi — slower, may overfit on 89 eps |

## data-side levers

| | move | yield estimate |
|---|---|---|
| A | record 30 more full successes (60→90) | mid — bigger success tail boosts succ_term |
| B | record dedicated stage 5→6 plateaus (the "tried and failed cover placement" mode) | hi — currently zero coverage, would directly target succ_term gate |
| C | record stage-1 (touched box but didn't grasp) failures | lo — already 5 in 1/6 bucket |

## ACT v3 plan (downstream of SARM v3)

### prereq
SARM v3 works (any branch above). Fresh CLIP features for 224×224 already built.

### iter1: stock v11 recipe on v3 success-only
- ds: `domrachev03/sim_3stage_v3_success` (60 ep, all 6/6, 224×224)
- cfg copy `act_3stage_v3_train.json` → `act_3stage_v3_iter1_train.json`
- chunk=20, 80k steps, ResNet18+ImageNet, MIN_MAX state+action

### iter2 (if iter1 < 40%)
- chunk=30 (longer than 6-stage, ~half the typical 263-frame ep)
- OR encoder swap: ResNet18 → DINOv2 (handles 224 input natively, better for fine details like cap edges)

### iter3 (if iter2 < 40%)
- destale_tail30 trick (proven v2 path B)
- evaluate at thr=0.85 too (v3 SARM may not consistently saturate at 0.95)

## HIL-SERL residual finetune (epic-54 / -50 in queue)

after both SARM v3 + ACT v3 ship, residual SAC train using:
- base ACT policy = act_3stage_v3_iterN
- reward = SARM v3
- 1h training → target ≥90% succ rate

cfg already exists: `sim_residual_3stage_v1_train.json` — needs path updates.

## kill / pause conditions

- 5 SARM iters fruitless after v3-iter1 → pause SARM, accept best, ship for ACT regardless
- ACT iter5+ fruitless → record more data
- HIL-SERL <70% after 2h training → pause, deep diagnostic

## results log

(populate as iters complete; keep iter6-iter9 v2 history below)

### v2 iter6→iter9 reference (production = iter8)

| iter | full max | succ_term | stage_ne | stage_nb | plat |
|---|---|---|---|---|---|
| iter6 | 0.93 | 0.07 | 0.79 | 0.60 | 0.17 |
| iter7 | 0.93 | 0.05 | 0.83 | 0.58 | 0.22 |
| iter8 ✓ | 0.94 | 0.10 | 0.83 | 0.54 | 0.19 |
| iter9 | 0.89 | 0.13 | 0.79 | 0.52 | 0.10 |

### v3 iters

| iter | result |
|---|---|
| v3-iter1 (collapsed; no-video patch + my modeling edits) | max=0.07, succ=0.00 — bug |
| v3-iter1 stock+reverted (current, 87% trained) | TBD |
