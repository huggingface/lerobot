# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v3_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v3_iter8/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v3_no0_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.399 | 0.250 | ❌ |
| mean_mid | 0.120 | 0.250 | ❌ |
| monotonicity | 0.511 | 0.850 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.000 | 0.000 | ✅ |
| plateau_ok_rate | 0.300 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.970 | 0.900 | ✅ |
| stage_not_below_rate | 0.241 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.10 | 0.31 | 0.00 | 0.00 | 0.00 | nan | 0.123 | nan | 0.620 | 1.000 | 0.00 |
| 1-of-6 | 5 | 0.11 | 0.24 | 0.00 | 0.00 | 0.00 | nan | 0.037 | 1.000 | 0.857 | 1.000 | 0.80 |
| 2-of-6 | 5 | 0.18 | 0.34 | 0.00 | 0.00 | 0.00 | nan | 0.166 | 0.451 | 0.980 | 0.480 | 1.00 |
| 3-of-6 | 5 | 0.09 | 0.24 | 0.00 | 0.00 | 0.00 | nan | 0.107 | 0.038 | 1.000 | 0.057 | 0.00 |
| 4-of-6 | 5 | 0.17 | 0.29 | 0.00 | 0.00 | 0.00 | nan | 0.111 | 0.023 | 1.000 | 0.062 | 0.00 |
| 5-of-6 | 5 | 0.05 | 0.27 | 0.00 | 0.00 | 0.00 | nan | 0.070 | 0.028 | 1.000 | 0.045 | 0.00 |
| full | 59 | 0.04 | 0.28 | 0.00 | 0.00 | 0.00 | 0.399 | 0.120 | 0.092 | 1.000 | 0.140 | nan |