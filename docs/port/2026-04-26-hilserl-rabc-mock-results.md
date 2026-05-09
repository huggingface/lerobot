# HIL-SERL RABC mock-test results (2026-04-26)

epic lerobot-50. validates new BC+RABC code path in SAC + 4 cfg variants
without launching teleop / actor / env. 10 opt steps each on synth batches.

## summary

| var | status | bc_w | rabc_mean | bc_loss | sac_loss | crit_loss | temp | params_upd |
|---|---|---|---|---|---|---|---|---|
| V1 | PASS | 0.0 | — | — | -6.77 | 0.61 | 0.997 | 26/26 |
| V2 | PASS | 0.5 | 0.79 | 2.76 | -7.77 | 0.71 | 0.997 | 26/26 |
| V3 | PASS | 0.5 | 0.79 | 2.75 | -7.54 | 0.48 | 0.997 | 26/26 |
| V4 | PASS | 0.5 | 0.83 | 2.64 | -7.19 | 0.41 | 0.997 | 26/26 |

acceptance gates ✓:
- ckpt load (V3/V4 read `outputs/bc_pretrain_v1/last`)
- BC fwd path active (loss_actor_bc populated when bc_w>0)
- RABC weights propagate (mean ∈ [0.79, 0.83]; zero+full counts logged)
- no NaN in any step
- 26/26 actor params updated (via grad)
- 10 steps in <30s per variant

## artifacts produced

```
src/lerobot/scripts/build_rabc_progress_from_delta.py  # progress parquet builder
src/lerobot/scripts/bc_pretrain_sac.py                 # BC pretrain entrypoint
src/lerobot/policies/sac/configuration_sac.py          # +bc_loss_weight, bc_use_rabc, ...
src/lerobot/policies/sac/modeling_sac.py               # compute_loss_actor +bc args
src/lerobot/rl/learner.py                              # bc inputs from offline batch
src/lerobot/rl/sim_assembling_sarm_hilserl_rabc_v{1..4}_train.json
outputs/bc_pretrain_v1/last/                           # 200-step quick BC pretrain ckpt (10 eps)
~/.cache/huggingface/lerobot/local/sim_assemble_sarm_merged_v1_sarm_delta/sarm_progress.parquet
```

## known mock limitations (non-blocking)

- BC pretrain ran 200 steps on 10 eps to seed V3/V4. real run = 5000 steps × full 103 eps before HIL-SERL.
- RABC `index` field missing in offline replay buffer's batch dict — RABCWeights falls back to
  uniform when called from `bc_pretrain_sac.py`. Need to thread global `index` through
  `ReplayBuffer.from_lerobot_dataset` for full RABC effect during pretraining.
  HIL-SERL learner integration (mock) DOES feed `index` because the synth batch sets it
  to real progress-parquet keys — production needs the same wiring in the live offline
  buffer batch (currently absent).
- `use_torch_compile=true` triggers dynamo on first batch (compiles the encoder against
  symbolic shapes). Mock disabled compile to avoid recompile overhead. real runs keep it on.
- temperature still ≈ 1.0 after only 10 steps; full 1h run needed to confirm temp_init=1
  + target_entropy=-4 actually prevents the entropy crash seen in `ciaktf45` /  `bzp2mtuc`.

## ready for live HIL-SERL smoke

next session checklist:
1. wire offline `index` → batch dict in `buffer.py` so RABC sees real frame ids
2. run full BC pretrain (5k steps, full ds) → fresh `outputs/bc_pretrain_v1/last`
3. 15-min HIL-SERL smoke per variant (start V2, then V3); compare wandb
4. if BC loss + temp stays sane → 1h run

## diagnostics from prior runs (recap)

| run | succ | ep_rwrd | intrv | temp_final | issue |
|---|---|---|---|---|---|
| ciaktf45 | 0/226 | 0.48 | 47.7% | 0.003 | stage1 plateau, entropy collapse |
| bzp2mtuc | early | 0.95 (w/ intrv) | 81% | 0.007 | almost all human |

V1 fixes entropy via temp_init=1 + target_ent=-4 + std_min=0.05. V2 adds BC+RABC pull
toward demos. V3 starts actor at demo-imitating mean. V4 sweeps discount/utd for
sparser-reward credit assignment.
