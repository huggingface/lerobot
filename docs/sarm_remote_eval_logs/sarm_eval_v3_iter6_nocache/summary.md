# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v3_val_fs`
- ckpt: `/home/dom_iva/github.com/orel/lerobot/lerobot/outputs/sim_3stage_sarm_v3_iter6/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v3_success_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.051 | 0.950 | ❌ |
| lin_mad | 0.107 | 0.250 | ✅ |
| mean_mid | 0.451 | 0.250 | ✅ |
| monotonicity | 0.862 | 0.850 | ✅ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 1.000 | 0.000 | ❌ |
| plateau_ok_rate | 0.100 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.769 | 0.900 | ❌ |
| stage_not_below_rate | 0.789 | 0.700 | ✅ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.40 | 0.86 | 0.00 | 0.00 | 1.00 | nan | 0.460 | nan | 0.105 | 1.000 | 0.00 |
| 1-of-6 | 5 | 0.30 | 0.64 | 0.00 | 0.00 | 0.80 | nan | 0.213 | 0.200 | 0.170 | 1.000 | 0.00 |
| 2-of-6 | 5 | 0.26 | 0.79 | 0.00 | 0.00 | 0.80 | nan | 0.303 | 0.263 | 0.357 | 0.925 | 0.00 |
| 3-of-6 | 5 | 0.37 | 0.45 | 0.00 | 0.00 | 0.00 | nan | 0.372 | 0.234 | 0.285 | 0.960 | 0.20 |
| 4-of-6 | 5 | 0.50 | 0.74 | 0.00 | 0.00 | 1.00 | nan | 0.469 | 0.470 | 0.705 | 0.785 | 0.40 |
| 5-of-6 | 5 | 0.51 | 0.88 | 0.00 | 0.00 | 1.00 | nan | 0.702 | 0.313 | 0.530 | 0.795 | 0.00 |
| full | 59 | 0.79 | 0.88 | 0.05 | 0.07 | 1.00 | 0.107 | 0.451 | 0.692 | 0.978 | 0.726 | nan |