# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v4_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v4_6stage_wrist_invfreq/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v4_success_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.412 | 0.250 | ❌ |
| mean_mid | 0.104 | 0.250 | ❌ |
| monotonicity | 0.546 | 0.850 | ❌ |
| last_stage_max_prog_rate | 0.000 | 1.000 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.600 | 0.000 | ❌ |
| plateau_ok_rate | 0.267 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.992 | 0.900 | ✅ |
| stage_not_below_rate | 0.150 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.04 | 0.48 | 0.00 | 0.00 | 0.60 | nan | 0.091 | nan | 0.890 | 1.000 | 0.00 |
| 1-of-6 | 5 | 0.04 | 0.35 | 0.00 | 0.00 | 0.40 | nan | 0.041 | 1.000 | 0.865 | 1.000 | 0.40 |
| 2-of-6 | 5 | 0.05 | 0.33 | 0.00 | 0.00 | 0.40 | nan | 0.062 | 0.077 | 0.967 | 0.133 | 0.40 |
| 3-of-6 | 5 | 0.04 | 0.34 | 0.00 | 0.00 | 0.40 | nan | 0.074 | 0.023 | 0.995 | 0.048 | 0.00 |
| 4-of-6 | 5 | 0.05 | 0.32 | 0.00 | 0.00 | 0.20 | nan | 0.068 | 0.008 | 0.988 | 0.055 | 0.20 |
| 5-of-6 | 5 | 0.04 | 0.43 | 0.00 | 0.00 | 0.60 | nan | 0.056 | 0.025 | 1.000 | 0.042 | 0.60 |
| full | 160 | 0.09 | 0.41 | 0.00 | 0.00 | 0.53 | 0.412 | 0.104 | 0.055 | 1.000 | 0.107 | nan |