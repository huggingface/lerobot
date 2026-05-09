# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v3_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v3_iter5/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v3_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.495 | 0.250 | ❌ |
| mean_mid | 0.003 | 0.250 | ❌ |
| monotonicity | 0.819 | 0.850 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.000 | 0.000 | ✅ |
| plateau_ok_rate | 0.167 | 0.800 | ❌ |
| stage_not_exceed_rate | 1.000 | 0.900 | ✅ |
| stage_not_below_rate | 0.154 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | nan | 0.000 | nan | 1.000 | 1.000 | 1.00 |
| 1-of-6 | 5 | 0.03 | 0.06 | 0.00 | 0.00 | 0.00 | nan | 0.008 | 1.000 | 1.000 | 1.000 | 0.00 |
| 2-of-6 | 5 | 0.02 | 0.04 | 0.00 | 0.00 | 0.00 | nan | 0.008 | 0.000 | 1.000 | 0.028 | 0.00 |
| 3-of-6 | 5 | 0.01 | 0.03 | 0.00 | 0.00 | 0.00 | nan | 0.011 | 0.000 | 1.000 | 0.020 | 0.00 |
| 4-of-6 | 5 | 0.01 | 0.03 | 0.00 | 0.00 | 0.00 | nan | 0.010 | 0.000 | 1.000 | 0.040 | 0.00 |
| 5-of-6 | 5 | 0.01 | 0.02 | 0.00 | 0.00 | 0.00 | nan | 0.004 | 0.000 | 1.000 | 0.017 | 0.00 |
| full | 59 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.495 | 0.003 | 0.000 | 1.000 | 0.053 | nan |