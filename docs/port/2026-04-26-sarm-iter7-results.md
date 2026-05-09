# SARM iter-7 results (2026-04-26)

beads lerobot-49. iter-7 = filtered + _2 + _3 plateau-only eps (5 × 3/4).

## comparison: iter5 → iter6 → iter7

note: iter-7 val_fs is merged_v3-stride (108 eps, 18/15/15/20/40); iter-5/6 val_fs was merged_v2-stride (129 eps, 28/15/26/20/40). different val sets, but same source mix + frame-stride convention; numbers comparable.

### full val (overall gates, async = HIL-SERL realistic)

| metric | iter5 async | iter6 async | iter7 async |
|---|---|---|---|
| succ_term @0.95 | 0.075 | 0.150 | 0.125 |
| succ_max  @0.95 | 0.075 | 0.175 | 0.150 |
| succ_term @0.90 | 0.175 | 0.575 | 0.575 |
| succ_max  @0.90 | 0.350 | **0.950** ✓ | **0.950** ✓ |
| 0/4 max≥0.5    | 0.393 | 0.250 | **0.333** ❌↓ |
| lin_mad        | 0.166 | 0.146 | **0.140** ✓ |
| mean_mid       | 0.561 | 0.625 | 0.607 |
| monotonicity   | 0.802 ❌ | 0.905 ✓ | **0.911** ✓ |
| plateau_ok     | 0.697 ❌ | 0.775 ❌ | **0.809** ✓ |
| stage_ne       | 0.965 | 0.993 | **0.989** ✓ |
| stage_nb       | 0.778 | 0.846 | **0.830** |

### per-bucket: full (4/4) bucket only — directly answers user's stage-flicker concern

| metric (async, full bucket) | iter5 | iter6 | iter7 |
|---|---|---|---|
| **stage_nb (full)** | n/a | **0.840** | **0.647** ❌↓↓ |
| plateau_ok | n/a | n/a | n/a |
| max | n/a | n/a | 0.92 |

**iter-7 full-bucket stage_nb dropped 0.84→0.65** = ~35% of full ep frames have pred_stage < gt_stage. **Worse stage flicker than iter-6, much worse than user's pain threshold.** ❌

### new-val (5 _3 plateau, 3/4 bucket — what iter-7 was DESIGNED to fix)

| metric (async) | iter5 | iter6 | iter7 |
|---|---|---|---|
| plateau_ok rate | 0% | 80% | **100%** ✓ |
| max ∈ [0.74, 0.94] (all 5) | no | no | **yes** ✓ |
| 0/4-style spike on plateau | yes | no | no |
| stage_nb on plateau | n/a | n/a | **0.89** ✓ |

iter-7 **perfectly handles** the 5 _3 plateau eps. peak between 0.84-0.88 → never spikes past 0.95 success threshold → stays a fail (correct).

## verdict

**iter-7 = local optimum, not global. ship NO. fall back to iter-5 stays.**

**why:** the targeted hypothesis worked — adding 5 plateau eps fixed plateau handling perfectly. but the 5 eps perturbed the decision boundary and hurt **full-ep stage_nb (async, 4/4 bucket)**: 0.84→0.65. Same gate iter-6 already had as a weak point; iter-7 makes it **worse**.

user's teleop pain (stage 2→1 regression mid-episode) is exactly stage_nb < 1.0 on full bucket. iter-7 makes that **2× worse than iter-6**.

**iter-7 ≼ iter-6 ≼ iter-5 on user-visible stage stability.** even though iter-7 wins on plateau gates, the fundamental flicker problem worsened.

## next moves (ranked)

| # | hypothesis | leverage | risk |
|---|---|---|---|
| 1 | **post-hoc monotonic stage hysteresis** at inference (no retrain) — once stage k committed, never report < k for the rest of episode | high; eliminates the user-visible flicker entirely | none |
| 2 | iter-8: train with 0/4-replicate-future augmentation (50% prob) — match async window distribution | medium-high; closes sync↔async gap | low |
| 3 | iter-8: bigger n_obs_steps 8→12 — more context → less flicker | medium; cost ↑ ~30% | low |
| 4 | abandon SARM, fall back to binary CNN reward classifier | high (different flicker mode) | high (different gates) |

**recommend:** #1 NOW (10-min code change, no retrain), retest in teleop. iter-7 ckpt + monotonic stage post-process likely matches iter-5 user-experience. if still flickering → #2/#3 as iter-8.

## artifacts
- ckpt: `outputs/sim_assemble_sarm_ext_iter7_2cam/checkpoints/last/pretrained_model`
- evals: `outputs/sarm_eval/iter7_val_{sync,async}/`
- train ds: `local/sim_assemble_sarm_merged_v3_train_fs` (108 eps, 25620 fr)

production stays on **iter-5**. iter-7 ckpt kept for inspection.
