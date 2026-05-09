# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v3_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v3_iter6/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v3_success_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.432 | 0.250 | ❌ |
| mean_mid | 0.113 | 0.250 | ❌ |
| monotonicity | 0.482 | 0.850 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.200 | 0.000 | ❌ |
| plateau_ok_rate | 0.300 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.940 | 0.900 | ✅ |
| stage_not_below_rate | 0.269 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.26 | 0.48 | 0.00 | 0.00 | 0.20 | nan | 0.153 | nan | 0.718 | 1.000 | 0.00 |
| 1-of-6 | 5 | 0.12 | 0.39 | 0.00 | 0.00 | 0.00 | nan | 0.074 | 0.400 | 0.613 | 1.000 | 0.00 |
| 2-of-6 | 5 | 0.17 | 0.48 | 0.00 | 0.00 | 0.40 | nan | 0.188 | 0.175 | 0.760 | 0.435 | 0.00 |
| 3-of-6 | 5 | 0.06 | 0.41 | 0.00 | 0.00 | 0.00 | nan | 0.156 | 0.023 | 0.853 | 0.190 | 0.60 |
| 4-of-6 | 5 | 0.27 | 0.39 | 0.00 | 0.00 | 0.00 | nan | 0.172 | 0.260 | 1.000 | 0.295 | 0.80 |
| 5-of-6 | 5 | 0.13 | 0.52 | 0.00 | 0.00 | 0.40 | nan | 0.094 | 0.013 | 0.995 | 0.035 | 0.40 |
| full | 59 | 0.01 | 0.40 | 0.00 | 0.00 | 0.03 | 0.432 | 0.113 | 0.107 | 1.000 | 0.155 | nan |