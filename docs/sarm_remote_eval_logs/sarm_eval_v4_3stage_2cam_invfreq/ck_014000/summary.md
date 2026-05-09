# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v4_3stage_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v4_3stage_2cam_invfreq/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v4_3stage_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.341 | 0.250 | ❌ |
| mean_mid | 0.215 | 0.250 | ❌ |
| monotonicity | 0.689 | 0.850 | ❌ |
| last_stage_max_prog_rate | 0.000 | 1.000 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.600 | 0.000 | ❌ |
| plateau_ok_rate | 0.280 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.981 | 0.900 | ✅ |
| stage_not_below_rate | 0.349 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-3 | 5 | 0.21 | 0.49 | 0.00 | 0.00 | 0.60 | nan | 0.250 | nan | 0.552 | 1.000 | 0.00 |
| 1-of-3 | 10 | 0.10 | 0.35 | 0.00 | 0.00 | 0.20 | nan | 0.138 | 0.900 | 0.871 | 1.000 | 0.50 |
| 2-of-3 | 10 | 0.08 | 0.37 | 0.00 | 0.00 | 0.20 | nan | 0.113 | 0.128 | 0.998 | 0.190 | 0.20 |
| full | 165 | 0.19 | 0.46 | 0.00 | 0.00 | 0.61 | 0.341 | 0.215 | 0.154 | 1.000 | 0.300 | nan |