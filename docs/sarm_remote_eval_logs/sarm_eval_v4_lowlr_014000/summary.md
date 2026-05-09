# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v4_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v4_lowlr/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v4_success_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.366 | 0.250 | ❌ |
| mean_mid | 0.152 | 0.250 | ❌ |
| monotonicity | 0.638 | 0.850 | ❌ |
| last_stage_max_prog_rate | 0.000 | 1.000 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.000 | 0.000 | ✅ |
| plateau_ok_rate | 0.333 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.959 | 0.900 | ✅ |
| stage_not_below_rate | 0.229 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.12 | 0.36 | 0.00 | 0.00 | 0.00 | nan | 0.185 | nan | 0.260 | 1.000 | 0.00 |
| 1-of-6 | 5 | 0.13 | 0.28 | 0.00 | 0.00 | 0.20 | nan | 0.122 | 0.200 | 0.277 | 1.000 | 0.60 |
| 2-of-6 | 5 | 0.13 | 0.38 | 0.00 | 0.00 | 0.00 | nan | 0.161 | 0.739 | 0.925 | 0.815 | 0.20 |
| 3-of-6 | 5 | 0.14 | 0.34 | 0.00 | 0.00 | 0.00 | nan | 0.151 | 0.066 | 0.980 | 0.105 | 1.00 |
| 4-of-6 | 5 | 0.16 | 0.39 | 0.00 | 0.00 | 0.00 | nan | 0.135 | 0.046 | 0.993 | 0.087 | 0.20 |
| 5-of-6 | 5 | 0.13 | 0.35 | 0.00 | 0.00 | 0.20 | nan | 0.148 | 0.036 | 1.000 | 0.053 | 0.00 |
| full | 160 | 0.15 | 0.33 | 0.00 | 0.00 | 0.16 | 0.366 | 0.152 | 0.129 | 1.000 | 0.176 | nan |