# SARM iter-7 plan (2026-04-26)

beads lerobot-49. follows iter-6 rollback.

## hypothesis
add only _3 3/4-plateau eps (5 eps, ~3000 fr) as hard-neg for succ/fail distinction. avoid the bulk of _3 (0/4 + 2/4) that hurt iter-6 stage stability.

## inputs
- merged_v1 = filtered (51 ep) + _2 (52 ep) = 103 ep, 25448 fr (already on disk)
- _3 plateau = local _3 eps 25-29 only (5 eps × 600 fr ≈ 3000 fr; bucket 3/4)

## merged_v3
108 eps. bucket: 0:18, 1:15, 2:15, 3:**20** (=15+5), 4:40.

## execute
1. `delete_episodes(_3, [0..24])` → `local/_sim_assemble_sarm_3_3of4` (5 eps, 25-29)
2. `merge([filtered, _2, _3_3of4])` → `local/sim_assemble_sarm_merged_v3_full`
3. frame-stride split (N=10) → `merged_v3_train_fs` + `merged_v3_val_fs`
4. write proportions for both
5. CLIP cache train ds
6. train cfg copy iter-6 → iter-7 (only diff: dataset.repo_id, output_dir, steps=5000 since smaller ds)
7. train (background, ETA ~25min w/ cache + bs=32)
8. eval sync + async on val_fs at thr {0.9, 0.95}
9. compare to iter-5 baseline
10. if gates pass + stage stability ok at teleop → iter-7 = new prod

## risk
the 5 _3 plateau eps share scene config / behaviour with _3-bulk that hurt iter-6. if iter-7 also flickers stage at teleop → revert to iter-5 final.
