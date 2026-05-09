# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v4_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v4_lowlr/checkpoints/007000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v4_success_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.380 | 0.250 | ❌ |
| mean_mid | 0.125 | 0.250 | ❌ |
| monotonicity | 0.615 | 0.850 | ❌ |
| last_stage_max_prog_rate | 0.000 | 1.000 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.200 | 0.000 | ❌ |
| plateau_ok_rate | 0.300 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.965 | 0.900 | ✅ |
| stage_not_below_rate | 0.206 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.13 | 0.34 | 0.00 | 0.00 | 0.20 | nan | 0.157 | nan | 0.330 | 1.000 | 0.00 |
| 1-of-6 | 5 | 0.14 | 0.50 | 0.00 | 0.00 | 0.80 | nan | 0.146 | 0.200 | 0.355 | 1.000 | 0.20 |
| 2-of-6 | 5 | 0.14 | 0.22 | 0.00 | 0.00 | 0.00 | nan | 0.154 | 0.798 | 0.990 | 0.808 | 1.00 |
| 3-of-6 | 5 | 0.15 | 0.20 | 0.00 | 0.00 | 0.00 | nan | 0.135 | 0.041 | 1.000 | 0.060 | 0.20 |
| 4-of-6 | 5 | 0.16 | 0.27 | 0.00 | 0.00 | 0.20 | nan | 0.146 | 0.021 | 0.990 | 0.065 | 0.20 |
| 5-of-6 | 5 | 0.14 | 0.29 | 0.00 | 0.00 | 0.20 | nan | 0.134 | 0.033 | 1.000 | 0.050 | 0.20 |
| full | 160 | 0.11 | 0.23 | 0.00 | 0.00 | 0.03 | 0.380 | 0.125 | 0.103 | 1.000 | 0.151 | nan |