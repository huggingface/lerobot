# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v3_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v3_iter13/checkpoints/007000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v3_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.458 | 0.250 | ❌ |
| mean_mid | 0.065 | 0.250 | ❌ |
| monotonicity | 0.675 | 0.850 | ❌ |
| last_stage_max_prog_rate | 0.000 | 1.000 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.000 | 0.000 | ✅ |
| plateau_ok_rate | 0.133 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.991 | 0.900 | ✅ |
| stage_not_below_rate | 0.188 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.04 | 0.21 | 0.00 | 0.00 | 0.00 | nan | 0.037 | nan | 0.895 | 1.000 | 0.00 |
| 1-of-6 | 5 | 0.04 | 0.16 | 0.00 | 0.00 | 0.00 | nan | 0.009 | 1.000 | 0.940 | 1.000 | 0.80 |
| 2-of-6 | 5 | 0.10 | 0.21 | 0.00 | 0.00 | 0.00 | nan | 0.064 | 0.236 | 1.000 | 0.255 | 0.00 |
| 3-of-6 | 5 | 0.01 | 0.20 | 0.00 | 0.00 | 0.00 | nan | 0.033 | 0.015 | 1.000 | 0.035 | 0.00 |
| 4-of-6 | 5 | 0.00 | 0.22 | 0.00 | 0.00 | 0.00 | nan | 0.060 | 0.005 | 1.000 | 0.045 | 0.00 |
| 5-of-6 | 5 | 0.01 | 0.19 | 0.00 | 0.00 | 0.00 | nan | 0.019 | 0.008 | 1.000 | 0.025 | 0.00 |
| full | 59 | 0.01 | 0.20 | 0.00 | 0.00 | 0.00 | 0.458 | 0.065 | 0.031 | 1.000 | 0.083 | nan |