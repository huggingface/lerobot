# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v3_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v3_iter2/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v3_no0_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.410 | 0.250 | ❌ |
| mean_mid | 0.124 | 0.250 | ❌ |
| monotonicity | 0.507 | 0.850 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.000 | 0.000 | ✅ |
| plateau_ok_rate | 0.300 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.971 | 0.900 | ✅ |
| stage_not_below_rate | 0.239 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.10 | 0.24 | 0.00 | 0.00 | 0.00 | nan | 0.142 | nan | 0.600 | 1.000 | 0.00 |
| 1-of-6 | 5 | 0.12 | 0.22 | 0.00 | 0.00 | 0.00 | nan | 0.060 | 1.000 | 0.897 | 1.000 | 1.00 |
| 2-of-6 | 5 | 0.13 | 0.28 | 0.00 | 0.00 | 0.00 | nan | 0.189 | 0.471 | 0.992 | 0.487 | 0.80 |
| 3-of-6 | 5 | 0.16 | 0.25 | 0.00 | 0.00 | 0.00 | nan | 0.083 | 0.038 | 1.000 | 0.057 | 0.00 |
| 4-of-6 | 5 | 0.17 | 0.25 | 0.00 | 0.00 | 0.00 | nan | 0.113 | 0.018 | 1.000 | 0.057 | 0.00 |
| 5-of-6 | 5 | 0.07 | 0.24 | 0.00 | 0.00 | 0.00 | nan | 0.074 | 0.025 | 1.000 | 0.042 | 0.00 |
| full | 59 | 0.03 | 0.25 | 0.00 | 0.00 | 0.00 | 0.410 | 0.124 | 0.087 | 1.000 | 0.136 | nan |