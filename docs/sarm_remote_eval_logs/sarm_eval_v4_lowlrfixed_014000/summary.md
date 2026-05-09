# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v4_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v4_lowlrfixed/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v4_success_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.348 | 0.250 | ❌ |
| mean_mid | 0.191 | 0.250 | ❌ |
| monotonicity | 0.645 | 0.850 | ❌ |
| last_stage_max_prog_rate | 0.000 | 1.000 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.000 | 0.000 | ✅ |
| plateau_ok_rate | 0.300 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.956 | 0.900 | ✅ |
| stage_not_below_rate | 0.245 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.17 | 0.31 | 0.00 | 0.00 | 0.00 | nan | 0.184 | nan | 0.190 | 1.000 | 0.00 |
| 1-of-6 | 5 | 0.14 | 0.28 | 0.00 | 0.00 | 0.00 | nan | 0.157 | 0.200 | 0.210 | 1.000 | 0.00 |
| 2-of-6 | 5 | 0.14 | 0.30 | 0.00 | 0.00 | 0.00 | nan | 0.179 | 0.760 | 0.915 | 0.845 | 0.80 |
| 3-of-6 | 5 | 0.12 | 0.28 | 0.00 | 0.00 | 0.00 | nan | 0.165 | 0.097 | 1.000 | 0.115 | 1.00 |
| 4-of-6 | 5 | 0.17 | 0.28 | 0.00 | 0.00 | 0.00 | nan | 0.140 | 0.034 | 0.995 | 0.072 | 0.00 |
| 5-of-6 | 5 | 0.14 | 0.31 | 0.00 | 0.00 | 0.00 | nan | 0.162 | 0.046 | 1.000 | 0.062 | 0.00 |
| full | 160 | 0.20 | 0.28 | 0.00 | 0.00 | 0.00 | 0.348 | 0.191 | 0.148 | 1.000 | 0.194 | nan |