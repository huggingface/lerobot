# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v4_3stage_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v4_3stage_wrist_nostate/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v4_3stage_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.500 | 0.250 | ❌ |
| mean_mid | 0.000 | 0.250 | ❌ |
| monotonicity | 1.000 | 0.850 | ✅ |
| last_stage_max_prog_rate | 0.000 | 1.000 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.000 | 0.000 | ✅ |
| plateau_ok_rate | 0.200 | 0.800 | ❌ |
| stage_not_exceed_rate | 1.000 | 0.900 | ✅ |
| stage_not_below_rate | 0.233 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-3 | 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | nan | 0.000 | nan | 1.000 | 1.000 | 1.00 |
| 1-of-3 | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | nan | 0.000 | 1.000 | 1.000 | 1.000 | 0.00 |
| 2-of-3 | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | nan | 0.000 | 0.000 | 1.000 | 0.068 | 0.00 |
| full | 165 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.500 | 0.000 | 0.000 | 1.000 | 0.173 | nan |