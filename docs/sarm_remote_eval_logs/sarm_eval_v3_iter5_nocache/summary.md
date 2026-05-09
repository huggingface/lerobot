# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v3_val_fs`
- ckpt: `/home/dom_iva/github.com/orel/lerobot/lerobot/outputs/sim_3stage_sarm_v3_iter5/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v3_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.136 | 0.950 | ❌ |
| lin_mad | 0.154 | 0.250 | ✅ |
| mean_mid | 0.584 | 0.250 | ✅ |
| monotonicity | 0.807 | 0.850 | ❌ |
| fail_term_rate | 0.000 | 0.000 | ✅ |
| zero_max_ge_0.5 | 0.000 | 0.000 | ✅ |
| plateau_ok_rate | 1.000 | 0.800 | ✅ |
| stage_not_exceed_rate | 0.955 | 0.900 | ✅ |
| stage_not_below_rate | 0.787 | 0.700 | ✅ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-of-6 | 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | nan | 0.000 | nan | 1.000 | 1.000 | 1.00 |
| 1-of-6 | 5 | 0.15 | 0.16 | 0.00 | 0.00 | 0.00 | nan | 0.085 | 1.000 | 1.000 | 1.000 | 1.00 |
| 2-of-6 | 5 | 0.32 | 0.32 | 0.00 | 0.00 | 0.00 | nan | 0.254 | 0.992 | 0.982 | 0.993 | 1.00 |
| 3-of-6 | 5 | 0.45 | 0.49 | 0.00 | 0.00 | 0.00 | nan | 0.420 | 0.952 | 0.975 | 0.965 | 1.00 |
| 4-of-6 | 5 | 0.66 | 0.67 | 0.00 | 0.00 | 1.00 | nan | 0.573 | 0.922 | 0.953 | 0.967 | 1.00 |
| 5-of-6 | 5 | 0.81 | 0.82 | 0.00 | 0.00 | 1.00 | nan | 0.722 | 0.890 | 0.940 | 0.945 | 1.00 |
| full | 59 | 0.86 | 0.93 | 0.14 | 0.20 | 1.00 | 0.154 | 0.584 | 0.619 | 0.944 | 0.690 | nan |