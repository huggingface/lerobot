# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v3_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v3_iter14/checkpoints/007000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v3_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.384 | 0.250 | ❌ |
| mean_mid | 0.195 | 0.250 | ❌ |
| monotonicity | 0.637 | 0.850 | ❌ |
| last_stage_max_prog_rate | 0.000 | 1.000 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 1.000 | 0.000 | ❌ |
| plateau_ok_rate | 0.267 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.967 | 0.900 | ✅ |
| stage_not_below_rate | 0.303 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.28 | 0.75 | 0.00 | 0.00 | 1.00 | nan | 0.075 | nan | 0.795 | 1.000 | 0.00 |
| 1-of-6 | 5 | 0.03 | 0.80 | 0.00 | 0.00 | 1.00 | nan | 0.025 | 1.000 | 0.865 | 1.000 | 0.00 |
| 2-of-6 | 5 | 0.18 | 0.74 | 0.00 | 0.00 | 1.00 | nan | 0.112 | 0.192 | 0.915 | 0.295 | 0.00 |
| 3-of-6 | 5 | 0.01 | 0.31 | 0.00 | 0.00 | 0.40 | nan | 0.004 | 0.033 | 0.995 | 0.057 | 0.40 |
| 4-of-6 | 5 | 0.01 | 0.77 | 0.00 | 0.00 | 1.00 | nan | 0.119 | 0.102 | 0.950 | 0.190 | 0.40 |
| 5-of-6 | 5 | 0.01 | 0.81 | 0.00 | 0.00 | 1.00 | nan | 0.160 | 0.059 | 0.957 | 0.118 | 0.80 |
| full | 59 | 0.20 | 0.78 | 0.00 | 0.00 | 1.00 | 0.384 | 0.195 | 0.182 | 0.994 | 0.231 | nan |