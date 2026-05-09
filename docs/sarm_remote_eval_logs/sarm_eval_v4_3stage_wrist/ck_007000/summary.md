# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v4_3stage_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v4_3stage_wrist/checkpoints/007000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v4_3stage_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.382 | 0.250 | ❌ |
| mean_mid | 0.121 | 0.250 | ❌ |
| monotonicity | 0.643 | 0.850 | ❌ |
| last_stage_max_prog_rate | 0.000 | 1.000 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.000 | 0.000 | ✅ |
| plateau_ok_rate | 0.240 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.991 | 0.900 | ✅ |
| stage_not_below_rate | 0.260 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-3 | 5 | 0.11 | 0.32 | 0.00 | 0.00 | 0.00 | nan | 0.192 | nan | 0.847 | 1.000 | 0.00 |
| 1-of-3 | 10 | 0.13 | 0.30 | 0.00 | 0.00 | 0.00 | nan | 0.136 | 0.900 | 0.903 | 1.000 | 0.60 |
| 2-of-3 | 10 | 0.15 | 0.33 | 0.00 | 0.00 | 0.00 | nan | 0.115 | 0.089 | 1.000 | 0.150 | 0.00 |
| full | 165 | 0.16 | 0.25 | 0.00 | 0.00 | 0.05 | 0.382 | 0.121 | 0.031 | 1.000 | 0.199 | nan |