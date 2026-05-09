# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v3_val_fs`
- ckpt: `/home/dom_iva/github.com/orel/lerobot/lerobot/outputs/sim_3stage_sarm_v3_iter4/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v3_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.169 | 0.950 | ❌ |
| lin_mad | 0.172 | 0.250 | ✅ |
| mean_mid | 0.575 | 0.250 | ✅ |
| monotonicity | 0.787 | 0.850 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.000 | 0.000 | ✅ |
| plateau_ok_rate | 1.000 | 0.800 | ✅ |
| stage_not_exceed_rate | 0.955 | 0.900 | ✅ |
| stage_not_below_rate | 0.753 | 0.700 | ✅ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | nan | 0.000 | nan | 1.000 | 1.000 | 1.00 |
| 1-of-6 | 5 | 0.15 | 0.16 | 0.00 | 0.00 | 0.00 | nan | 0.090 | 1.000 | 1.000 | 1.000 | 1.00 |
| 2-of-6 | 5 | 0.32 | 0.33 | 0.00 | 0.00 | 0.00 | nan | 0.254 | 0.997 | 0.985 | 0.997 | 1.00 |
| 3-of-6 | 5 | 0.48 | 0.49 | 0.00 | 0.00 | 0.00 | nan | 0.421 | 0.949 | 0.978 | 0.962 | 1.00 |
| 4-of-6 | 5 | 0.66 | 0.66 | 0.00 | 0.00 | 1.00 | nan | 0.573 | 0.950 | 0.975 | 0.970 | 1.00 |
| 5-of-6 | 5 | 0.80 | 0.81 | 0.00 | 0.00 | 1.00 | nan | 0.724 | 0.901 | 0.937 | 0.952 | 1.00 |
| full | 59 | 0.86 | 0.94 | 0.17 | 0.27 | 1.00 | 0.172 | 0.575 | 0.558 | 0.943 | 0.637 | nan |