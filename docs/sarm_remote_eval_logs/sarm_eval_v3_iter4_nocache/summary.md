# SARM eval summary

## Config

- dataset: `domrachev03/sim_3stage_v3_val_fs`
- ckpt: `/home/dom_iva/github.com/orel/lerobot/lerobot/outputs/sim_3stage_sarm_v3_iter4/checkpoints/014000/pretrained_model`
- task: `Three-stage assembly`
- stats: `domrachev03/sim_3stage_v3_train_fs`

## Gates

| gate | value | threshold | pass |
|---|---|---|---|
| succ_term_rate | 0.600 | 0.950 | ❌ |
| lin_mad | 0.139 | 0.250 | ✅ |
| mean_mid | 0.607 | 0.250 | ✅ |
| monotonicity | 0.833 | 0.850 | ❌ |
| stage_not_exceed_rate | 0.952 | 0.900 | ✅ |
| stage_not_below_rate | 0.677 | 0.700 | ❌ |

## Per-bucket

| bucket | n | term | max | term≥thr | max≥thr | max≥0.5 | lin_mad | mean_mid | stage_argmax | stage_ne↑#1 | stage_nb↑#3 | plateau_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 5 | 0.94 | 0.95 | 0.60 | 0.60 | 1.00 | 0.139 | 0.607 | 0.618 | 0.952 | 0.677 | nan |