# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v3_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v3_iter3/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v3_no01_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.427 | 0.250 | ❌ |
| mean_mid | 0.073 | 0.250 | ❌ |
| monotonicity | 0.533 | 0.850 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.400 | 0.000 | ❌ |
| plateau_ok_rate | 0.267 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.939 | 0.900 | ✅ |
| stage_not_below_rate | 0.253 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.08 | 0.38 | 0.00 | 0.00 | 0.40 | nan | 0.115 | nan | 0.332 | 1.000 | 0.00 |
| 1-of-6 | 5 | 0.06 | 0.19 | 0.00 | 0.00 | 0.00 | nan | 0.035 | 0.800 | 0.590 | 1.000 | 0.40 |
| 2-of-6 | 5 | 0.10 | 0.18 | 0.00 | 0.00 | 0.00 | nan | 0.126 | 0.648 | 0.995 | 0.657 | 0.80 |
| 3-of-6 | 5 | 0.08 | 0.17 | 0.00 | 0.00 | 0.00 | nan | 0.085 | 0.038 | 1.000 | 0.057 | 0.00 |
| 4-of-6 | 5 | 0.08 | 0.37 | 0.00 | 0.00 | 0.40 | nan | 0.092 | 0.021 | 0.990 | 0.065 | 0.40 |
| 5-of-6 | 5 | 0.07 | 0.35 | 0.00 | 0.00 | 0.40 | nan | 0.067 | 0.033 | 1.000 | 0.050 | 0.00 |
| full | 59 | 0.03 | 0.18 | 0.00 | 0.00 | 0.02 | 0.427 | 0.073 | 0.094 | 1.000 | 0.142 | nan |