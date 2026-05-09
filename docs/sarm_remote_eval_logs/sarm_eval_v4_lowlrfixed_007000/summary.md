# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v4_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v4_lowlrfixed/checkpoints/007000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v4_success_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.348 | 0.250 | ❌ |
| mean_mid | 0.207 | 0.250 | ❌ |
| monotonicity | 0.622 | 0.850 | ❌ |
| last_stage_max_prog_rate | 0.000 | 1.000 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.400 | 0.000 | ❌ |
| plateau_ok_rate | 0.333 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.956 | 0.900 | ✅ |
| stage_not_below_rate | 0.239 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.18 | 0.42 | 0.00 | 0.00 | 0.40 | nan | 0.213 | nan | 0.165 | 1.000 | 0.00 |
| 1-of-6 | 5 | 0.14 | 0.40 | 0.00 | 0.00 | 0.40 | nan | 0.131 | 0.200 | 0.287 | 1.000 | 0.00 |
| 2-of-6 | 5 | 0.17 | 0.39 | 0.00 | 0.00 | 0.20 | nan | 0.191 | 0.754 | 0.907 | 0.847 | 0.40 |
| 3-of-6 | 5 | 0.18 | 0.34 | 0.00 | 0.00 | 0.00 | nan | 0.160 | 0.107 | 0.993 | 0.133 | 1.00 |
| 4-of-6 | 5 | 0.18 | 0.39 | 0.00 | 0.00 | 0.20 | nan | 0.142 | 0.033 | 0.990 | 0.077 | 0.20 |
| 5-of-6 | 5 | 0.14 | 0.44 | 0.00 | 0.00 | 0.40 | nan | 0.155 | 0.041 | 1.000 | 0.057 | 0.40 |
| full | 160 | 0.17 | 0.43 | 0.00 | 0.00 | 0.49 | 0.348 | 0.207 | 0.140 | 1.000 | 0.187 | nan |