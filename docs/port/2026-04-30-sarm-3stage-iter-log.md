# SARM 3-stage iter log

epic lerobot-60. start 2026-04-30. goal ≥90% gates on full val + new (success only first; partials if needed).

## dataset

src `domrachev03/sim_assemble_sarm_multistage_three_stages_success` (100 success eps, 26335 fr, 6 stages: approach_box, bring_box, approach_target, place_target_in_the_box, approach_cover, place_cover_on_the_box).

split: `scripts_local/frame_stride_split_3stage.py` N=10 → train_fs + val_fs.

## iter1 plan: success-only

| | val |
|---|---|
| cfg | `sim_3stage_sarm_train.json` (repo_id → train_fs) |
| arch | sarm_ext, dual head, 2cam, n_obs=8, gap=5, 7-D state |
| steps | 8k |
| output | outputs/sim_3stage_sarm/checkpoints/last |

eval cfg: same as v6 SARM eval harness `lerobot_policy_sarm.eval_sarm_sim_assemble`.
gates: stage_not_exceed (priority 1) > linearity > stage_not_below; plus plateau gates, monotonicity, mean_mid.

eval datasets:
- val_fs (in-distribution)
- old 2-stage success: `local/sim_assemble_sarm_merged_v1` (40 ep)
- old 2-stage failure: subset of original failure ds (TBD path)

## iter1 results

(eval running — populate when done)

## training-side improvements applied

| | impact |
|---|---|
| pre-encoded CLIP cache (`scripts_local/build_clip_cache.py`) | 2.4× speedup |
| batch_size 16→32 (top + policy + clip_batch_size) | 2× sample throughput / step |
| num_workers 4→16 | enough; 32 didn't help further (GPU saturated at 90%) |
| Iter1 wall-clock | ~30 min (vs initial ETA 70 min) |

## iter2 plan (regardless of iter1 — partials likely needed)

new dataset: `domrachev03/sim_assemble_sarm_multistage_three_stages_failures` — **19 partial-success eps** (verified labels: 9× 4/6, 10× 5/6, all len=800 timeout).

steps:
1. merge 100-success + 19 partial-fail → `domrachev03/sim_3stage_with_partials`.
2. frame-stride split (N=10) → train_partials_fs + val_partials_fs.
3. rebuild CLIP caches.
4. write temporal_proportions.
5. train iter2 (8k steps, same arch).
6. eval on val_partials_fs + new partials only + old success (val_fs from iter1).

note: old 2-stage datasets have incompatible stage names (2-stage vs 3-stage labels), **not merged**.

## hyp ranking (post-iter1, pre-iter2)

| # | hyp | leverage |
|---|---|---|
| 1 | partial-fail eps fix stage_not_below on stage 4-5 | hi (2-stage iter6 precedent) |
| 2 | n_obs_steps 8→12 longer context | mid |
| 3 | dense annotation mode | low |

## historical references (apply lessons to 3-stage)

- `docs/port/2026-04-25-sarm-ext-iters-findings.md` — 2-stage iter1-5: success-only on filtered_v2 → champion iter5. Established gate priority **stage_not_exceed > linearity > stage_not_below**.
- `docs/port/2026-04-26-sarm-iter6-results.md` — added _3 partial-fail eps (10× 0/4 fail + 5× 3/4 plateau). Result: better plateau gates BUT stage_nb on full-success regressed. Lesson: partial-fail injection is double-edged.
- `docs/port/2026-04-26-sarm-iter7-results.md` — iter7 fixed plateau gates perfectly but **made stage_nb worse**. Verdict: roll back to iter5 production. Recommendation: post-hoc monotonic stage hysteresis at inference (#1 of "next moves").
- `feedback_avoid_two_stages_3_ds.md` (memory) — production = `sim_assemble_sarm_ext_iter5_2cam` + `merged_v1` stats; user explicitly preferred iter5 over _3-trained iter6 due to teleop stage flicker.

**actionable lessons for 3-stage iter2 with new partials:**
1. validate stage_not_exceed (rank 1) doesn't drop on full-success eps when adding partials.
2. consider per-stage class weights to prevent stage 4-5 from over-shifting decision boundary.
3. iter6 used 10+5=15 partials → moderate effect. Our 19 partials is similar scale.
4. if iter2 hurts stage_nb on full eps, follow iter7 verdict: **prefer monotonic-stage-hysteresis post-hoc** over more partial-injection iters.

## hyp ranking (a priori)

| # | hyp | leverage |
|---|---|---|
| 1 | success-only is enough for 6-stage | mid-low (more stages → harder boundary) |
| 2 | need partials for stage 4-5 (cover) | hi |
| 3 | n_obs_steps 8→12 (longer context) | mid |
| 4 | dense annotation mode helps | low |
