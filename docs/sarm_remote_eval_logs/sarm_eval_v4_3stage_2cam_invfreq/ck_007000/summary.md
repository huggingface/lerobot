# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v4_3stage_val_fs`
- ckpt: `outputs/sim_3stage_sarm_v4_3stage_2cam_invfreq/checkpoints/007000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v4_3stage_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.000 | 0.950 | ❌ |
| lin_mad | 0.332 | 0.250 | ❌ |
| mean_mid | 0.232 | 0.250 | ❌ |
| monotonicity | 0.718 | 0.850 | ❌ |
| last_stage_max_prog_rate | 0.000 | 1.000 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.400 | 0.000 | ❌ |
| plateau_ok_rate | 0.240 | 0.800 | ❌ |
| stage_not_exceed_rate | 0.978 | 0.900 | ✅ |
| stage_not_below_rate | 0.349 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-3 | 5 | 0.21 | 0.47 | 0.00 | 0.00 | 0.40 | nan | 0.269 | nan | 0.535 | 1.000 | 0.00 |
| 1-of-3 | 10 | 0.14 | 0.37 | 0.00 | 0.00 | 0.10 | nan | 0.154 | 0.800 | 0.820 | 1.000 | 0.40 |
| 2-of-3 | 10 | 0.09 | 0.35 | 0.00 | 0.00 | 0.10 | nan | 0.134 | 0.172 | 0.999 | 0.230 | 0.20 |
| full | 165 | 0.20 | 0.53 | 0.00 | 0.00 | 0.86 | 0.332 | 0.232 | 0.150 | 1.000 | 0.296 | nan |