# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v3_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v3_seed1234/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v3_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.449 | 0.250 | ❌ |
| mean_mid | 0.083 | 0.250 | ❌ |
| monotonicity | 0.515 | 0.850 | ❌ |
| last_stage_max_prog_rate | 0.000 | 1.000 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.000 | 0.000 | ✅ |
| plateau_ok_rate | 0.167 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.995 | 0.900 | ✅ |
| stage_not_below_rate | 0.212 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.00 | 0.15 | 0.00 | 0.00 | 0.00 | nan | 0.005 | nan | 0.967 | 1.000 | 0.20 |
| 1-of-6 | 5 | 0.06 | 0.31 | 0.00 | 0.00 | 0.20 | nan | 0.014 | 1.000 | 0.953 | 1.000 | 0.80 |
| 2-of-6 | 5 | 0.04 | 0.19 | 0.00 | 0.00 | 0.00 | nan | 0.039 | 0.111 | 0.995 | 0.135 | 0.00 |
| 3-of-6 | 5 | 0.02 | 0.18 | 0.00 | 0.00 | 0.00 | nan | 0.004 | 0.036 | 1.000 | 0.055 | 0.00 |
| 4-of-6 | 5 | 0.00 | 0.19 | 0.00 | 0.00 | 0.00 | nan | 0.044 | 0.013 | 1.000 | 0.053 | 0.00 |
| 5-of-6 | 5 | 0.02 | 0.18 | 0.00 | 0.00 | 0.00 | nan | 0.003 | 0.020 | 1.000 | 0.037 | 0.00 |
| full | 59 | 0.01 | 0.20 | 0.00 | 0.00 | 0.00 | 0.449 | 0.083 | 0.078 | 1.000 | 0.127 | nan |