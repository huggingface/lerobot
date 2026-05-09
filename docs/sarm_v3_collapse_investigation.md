# SARM v3 Collapse Investigation

Started 2026-05-01. Status: open. Mandate: iterate till v2 iter8 trains; iter4-style v3 SARM works.

## Problem

v3 SARM train (6-stage assembly, `domrachev03/sim_3stage_v3_train_fs`, 89 eps, 36737 frames @ 224x224) collapses. Train loss healthy (~0.001 by 14k, matches iter4 to 3 sig figs). Eval: progress~0, predicted stage stuck at 0, all priority gates fail.

Only iter4 + iter5 worked (ckpts deleted). 25+ reproductions all collapse.

## Datasets

- `sim_3stage_v3_train_fs` 89 eps × 36737 fr, 2-cam 224x224, 6 sparse+dense stages. parquet md5 `ca15686481408a23c91c693a8b7d5e6e`. Stages: `approach_box, bring_box, approach_target, place_target_in_the_box, approach_cover, place_cover_on_the_box`. 100% temporal coverage.
- `sim_3stage_v3_val_fs` 89 eps × 4051 fr (frame-stride N=10).
- `sim_3stage_v3_no0_train_fs` 84 eps (drop 0/6).
- `sim_3stage_v3_no01_train_fs` 79 eps (drop 0+1/6).
- `sim_3stage_v3_success_train_fs` 59 eps (success only).

CLIP cache `meta/clip_cache.npz` md5 `c49ef47a6866ac120257a131d4c83eea`. Verified byte-identical local↔remote, cosine sim 1.0 vs fresh encode.

## Working baseline iter4

- Log: `/tmp/v3_iter4_remote.log` (DL_A6000, 2026-05-01 04:59)
- Recipe: sw=3.0, gt=0.75, max_rewind=3, n_obs=8, frame_gap=5, batch=32, workers=4, seed=1000, 14k steps, full data, dual mode
- Eval (nocache): succ_term=0.169 @0.95, mean_mid=0.575, max=0.94, lin_mad=0.172, mono=0.787, plateau=1.000, ne=0.955, nb=0.753 — passes 7/9 gates incl all 3 priority
- Per-bucket linear: 0/6→0.0, 1/6→0.16, 2/6→0.32, 3/6→0.49, 4/6→0.66, 5/6→0.81, full→0.86
- Ckpt deleted. Eval at `outputs/sarm_eval_v3_iter4_nocache_full/`.

## Reproduction failures (25)

| run | host | seed | sw | gt | rewind | data | extras | mean_mid |
|-----|------|------|-----|-----|--------|------|--------|----------|
| iter1 | local | 1000 | 1.0 | 0.75 | 3 | full | — | 0.008 |
| iter4 | remote | 1000 | 3.0 | 0.75 | 3 | full | — | **0.575 WORKED (lost)** |
| iter5 | remote | 1000 | 3.0 | 0.25 | 3 | full | — | **0.584 WORKED (lost)** |
| iter6 | remote | 1000 | 1.0 | 0.75 | 3 | success-only | — | 0.451 degraded |
| iter7 | remote | 1000 | 3.0 | 0.75 | 3 | full | 24k | 0.008 |
| iter8 | remote | 1000 | 3.0 | 0.75 | 3 | no0 | — | 0.120 |
| iter9 | remote | 1000 | 3.0 | 0.75 | 3 | no01 | — | 0.168 |
| iter10 | remote | 1000 | 1.0 | 0.75 | 3 | full | — | — |
| iter13 | remote | 1000 | 3.0 | 0.75 | 3 | full | tx 5.5.4 | 0.065 |
| iter14 | remote | 1000 | 3.0 | 0.25 | 3 | full | tx 5.5.4 | 0.195 |
| iter15 | local | 1000 | 3.0 | 0.75 | 3 | full | fast script | 0.006 |
| iter15 reencoded | local | 1000 | 3.0 | 0.75 | 3 | reencoded | fresh CLIP | 0.033 |
| iter15 fix | local | 1000 | 3.0 | 0.75 | 3 | full | per-sample rewind+masked loss | 0.075 |
| seed42 | remote | 42 | 3.0 | 0.75 | 3 | full | — | 0.013 |
| seed1234 | remote | 1234 | 3.0 | 0.75 | 3 | full | — | 0.083 |
| seed7777 | remote | 7777 | 3.0 | 0.75 | 3 | full | — | 0.011 |
| norewind | local | 1000 | 3.0 | 0.75 | **0** | full | — | 0.063 |
| classw | local | 1000 | 3.0 | 0.75 | 3 | full | inverse_freq | 0.004 |
| gt0 | remote | 1000 | 3.0 | **0.0** | 3 | full | — | 0.009 |
| gt1 | remote | 1000 | 3.0 | **1.0** | 3 | full | — | 0.006 |
| sw10 | remote | 1000 | **10.0** | 0.5 | 3 | full | — | 0.003 |
| bs16 | local | 1000 | 3.0 | 0.75 | 3 | full | bs=16 | 0.005 |
| norewind v2 | local | 1000 | 3.0 | 0.75 | 0 | full | + fixes | 0.077 |
| diag (fix) | local | 1000 | 3.0 | 0.75 | 3 | full | per-sample+masked | 0.075 |
| v2_iter8_repro | local | 1000 | 1.0 | 0.75 | 3 | v2_no01 | sanity check | running |

## Verifications

1. Data integrity: images sensible, 100% coverage, md5 stable.
2. CLIP cache: md5 identical local↔remote, cosine sim 1.0 fresh.
3. Code: rsync identical local↔remote. Patches active during iter4 (log shows `gt_stage_ratio:0.75` from my cfg field).
4. Transformers: 5.5.4 + 5.7.0 both collapse.
5. Eval pipeline: v2 iter8 numerical results identical old/new pipeline. Reproducible.
6. Loss curves: iter4 vs iter13/14 match 3 sig figs. Same loss, different model.
7. iter4 plots: smooth monotonic progress curves. Real working model, not eval bug.
8. Re-encoded v3: byte-identical to backup (lengths, annotations, action+state 0 abs diff). Not a data corruption.

## Diag instrumentation findings

Added stage entropy + argmax dist + gt dist logging per 50 steps in modeling_sarm.py + lerobot_train.py.

Found: 21% of train batches have **gt_frac0=1.000** (100% stage targets = class 0). Combined with low entropy → model memorizes per-batch GT. Trivial-stage-0 attractor wins because many batches are all stage 0.

## Root cause #1 (confirmed bug)

In processor_sarm.py:
```python
apply_rewind = self.training and random.random() < self.config.rewind_probability
```
Single per-batch decision, NOT per-sample. With rewind_p=0.8, 20% batches have rewind off for all 32 samples. Padded positions decode to stage=0,tau=0 → batches strongly stage-0-dominated.

## Fixes applied

- processor_sarm.py: per-sample rewind decision
- modeling_sarm.py: mask loss to valid frames (exclude padded positions)
- modeling_sarm.py: configurable gt_stage_ratio
- modeling_sarm.py: class_weights wired into F.cross_entropy
- eval: last_stage_max_prog_rate gate
- lerobot_train.py: SARM_DIAG line per log_freq

## Result of fixes

- iter15 fixed: 0.005 → 0.075 (10x). Still collapsed vs 0.575 iter4.
- norewind v2 (max_rewind=0 + fixes): 0.077. Same as orig 0.063.
- diag fix: 0.075. Per-batch homogeneity persists at lower frequency.

Per-batch fix helps marginally but doesn't escape collapse basin.

## Sanity check now running

v2_iter8_repro: known-working v2 recipe (sw=1.0, sim_3stage_v2_no01_train_fs) + current code (with fixes). 14k ETA ~36min.

- v2 collapses too → current code regression. Iterate till v2 trains.
- v2 works → v3-specific (sampler/data distribution).

## Hypothesis

Training landscape has 2 basins:
- A) "Meaningful": stage tracks GT order, progress 0→1
- B) "Trivial": predict stage 0, tau~0, low CE because rewind+padding produce many stage-0 targets

iter4/5 in A. 25+ reproductions in B. Loss equivalent in both. lerobot DataLoader worker_init_fn not deterministically seeded → batch order differs even with seed=1000. iter4 lucky early batches.

## Tried + ruled out

- All seeds (1000, 42, 1234, 7777) collapse
- sw 1.0, 3.0, 10.0 all collapse
- gt 0.0, 0.25, 0.5, 0.75, 1.0 all collapse
- rewind disabled collapses
- inverse_freq class weights collapse
- transformers 5.5.4 + 5.7.0 both fail
- local 4070 Ti + remote A6000 both fail
- vanilla lerobot-train + sarm_train_no_video.py both fail
- batch_size 16, 32 collapse (128 had startup issue)
- Re-encode dataset from sources: identical result
- Per-sample rewind fix: marginal improvement
- Loss masking padded positions: marginal improvement

## Future experiments

- Warm start from v2 iter8 ckpt (deferred per user)
- Big seed sweep (10-20 seeds)
- Different LR (peak_lr=1e-4 or 5e-6)
- SGD instead of AdamW
- Disable warmup
- Constant LR (no cosine decay)
- Smaller hidden_dim
- Single-cam ablation (front only)
- CNN classifier (`sim_assembling_cnn_reward_train_v3.yaml`)

## Key paths

- Cfgs: `src/lerobot/rl/sim_3stage_sarm_v3_*_train.json`
- Code: `lerobot_policy_sarm/src/lerobot_policy_sarm/{configuration,modeling,processor,eval_sarm_sim_assemble}.py`
- Fast script: `scripts_local/sarm_train_no_video.py`
- Eval outputs: `outputs/sarm_eval_v3_*` local + DL_A6000:`/home/dom_iva/.../outputs/`
- Train logs: local `/tmp/v3_*.log`, remote `DL_A6000:/tmp/v3_*.log`

## Commits

- lerobot_policy_sarm @ master 353bb81: per-sample rewind, masked loss, diag stats, last_stage_max_prog gate, gt_stage_ratio configurable, class_weights wired
- lerobot @ feature/reward-models-port 7503e086: SARM_DIAG log line
