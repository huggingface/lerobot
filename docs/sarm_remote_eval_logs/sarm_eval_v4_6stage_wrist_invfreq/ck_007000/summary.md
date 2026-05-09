# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v4_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v4_6stage_wrist_invfreq/checkpoints/007000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v4_success_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.433 | 0.250 | ❌ |
| mean_mid | 0.092 | 0.250 | ❌ |
| monotonicity | 0.526 | 0.850 | ❌ |
| last_stage_max_prog_rate | 0.000 | 1.000 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.400 | 0.000 | ❌ |
| plateau_ok_rate | 0.400 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.995 | 0.900 | ✅ |
| stage_not_below_rate | 0.108 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.05 | 0.34 | 0.00 | 0.00 | 0.40 | nan | 0.089 | nan | 0.923 | 1.000 | 0.20 |
| 1-of-6 | 5 | 0.05 | 0.14 | 0.00 | 0.00 | 0.00 | nan | 0.044 | 1.000 | 0.907 | 1.000 | 1.00 |
| 2-of-6 | 5 | 0.05 | 0.29 | 0.00 | 0.00 | 0.20 | nan | 0.064 | 0.010 | 0.982 | 0.055 | 0.20 |
| 3-of-6 | 5 | 0.04 | 0.20 | 0.00 | 0.00 | 0.20 | nan | 0.062 | 0.000 | 0.997 | 0.023 | 0.00 |
| 4-of-6 | 5 | 0.06 | 0.22 | 0.00 | 0.00 | 0.20 | nan | 0.052 | 0.000 | 0.995 | 0.045 | 0.20 |
| 5-of-6 | 5 | 0.05 | 0.52 | 0.00 | 0.00 | 0.80 | nan | 0.071 | 0.015 | 1.000 | 0.032 | 0.80 |
| full | 160 | 0.06 | 0.35 | 0.00 | 0.00 | 0.41 | 0.433 | 0.092 | 0.008 | 1.000 | 0.061 | nan |