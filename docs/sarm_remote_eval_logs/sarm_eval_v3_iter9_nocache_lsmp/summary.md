# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v3_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v3_iter9/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v3_no01_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.388 | 0.250 | ❌ |
| mean_mid | 0.168 | 0.250 | ❌ |
| monotonicity | 0.565 | 0.850 | ❌ |
| last_stage_max_prog_rate | 0.017 | 1.000 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 1.000 | 0.000 | ❌ |
| plateau_ok_rate | 0.267 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.910 | 0.900 | ✅ |
| stage_not_below_rate | 0.315 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.29 | 0.78 | 0.00 | 0.00 | 1.00 | nan | 0.231 | nan | 0.267 | 1.000 | 0.00 |
| 1-of-6 | 5 | 0.15 | 0.80 | 0.00 | 0.00 | 1.00 | nan | 0.167 | 0.800 | 0.410 | 1.000 | 0.00 |
| 2-of-6 | 5 | 0.09 | 0.60 | 0.00 | 0.00 | 1.00 | nan | 0.185 | 0.591 | 0.915 | 0.680 | 0.00 |
| 3-of-6 | 5 | 0.12 | 0.66 | 0.00 | 0.00 | 0.80 | nan | 0.144 | 0.049 | 0.917 | 0.150 | 0.20 |
| 4-of-6 | 5 | 0.26 | 0.64 | 0.00 | 0.00 | 0.80 | nan | 0.135 | 0.069 | 0.952 | 0.150 | 0.40 |
| 5-of-6 | 5 | 0.04 | 0.84 | 0.00 | 0.00 | 1.00 | nan | 0.283 | 0.233 | 0.953 | 0.293 | 1.00 |
| full | 59 | 0.10 | 0.70 | 0.00 | 0.02 | 0.88 | 0.388 | 0.168 | 0.151 | 0.998 | 0.197 | nan |