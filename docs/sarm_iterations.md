# SARM iteration log (lerobot-123)

Eval: `domrachev03/sim_3stage_v2_val_fs` (158 eps: 100 full + 9-10 each of 0/6..5/6 buckets, 128x128).
10-gate eval via `python -m lerobot_policy_sarm.eval_sarm_sim_assemble`.

Priority: stage_not_exceed (#1) > linearity > stage_not_below (#3). Reject if zero_max_ge_0.5 high.

## Gates (passing thresholds)
| gate | thr | direction |
|---|---|---|
| succ_term_rate | ≥0.95 | full eps' terminal ≥ 0.95 |
| lin_mad | ≤0.25 | full eps' deviation from linear |
| mean_mid | ≥0.25 | mid-time prog |
| monotonicity | ≥0.85 | non-decreasing curve |
| last_stage_max_prog_rate | ≥1.0 | full eps reach last stage |
| fail_term_rate | ≤0.0 | partial eps DON'T finish high |
| zero_max_ge_0.5 | ≤0.0 | 0-stage eps stay LOW |
| plateau_ok_rate | ≥0.8 | partials' peak in plateau zone |
| stage_not_exceed_rate | ≥0.9 | predicted stage ≤ GT stage |
| stage_not_below_rate | ≥0.7 | predicted stage ≥ GT stage |

## Results
| name | dataset | n_eps | gates_pass | succ_term | mean_mid | mono | zero_max | s_n_exceed | s_n_below | last_stage |
|---|---|---|---|---|---|---|---|---|---|---|
| v2_nostale_2cam (PROD) | v2_no01 | 139 (85s+54p) | 3/10 | 0.12 | 0.58 | 0.72 | 1.00 | 0.83 | 0.55 | 0.71 |
| v2_full_nostale_2cam | v2_full = no01 + extra | 288 | TBD | - | - | - | - | - | - | - |
| v2_full_v2_nostale_2cam (mixed sw3) | v2_train_fs + extra | 307 (205s+102p) | **5/10** | 0.06 | 0.43 | 0.67 | 0.89 | **0.945** | 0.42 | 0.75 |
| v2_full_v2_succonly_sw3 | v2_train_fs+extra succ-only | 205 | 3/10 | 0.17 | 0.45 | 0.70 | 1.00 | 0.81 | 0.57 | 0.38 |
| v2_full_v2_succonly_sw5 | v2_train_fs+extra succ-only | 205 | 3/10 | 0.19 | 0.47 | 0.72 | 1.00 | 0.81 | 0.61 | 0.47 |
| v2_full_v2_succonly_sw10 | v2_train_fs+extra succ-only | 205 | 3/10 | 0.16 | 0.45 | 0.71 | 1.00 | 0.82 | 0.57 | 0.45 |
| v2_full_v2_mixed_sw10 | v2_train_fs+extra | 307 | **5/10** | 0.09 | 0.44 | 0.65 | 0.67 | **0.975** | 0.40 | 0.76 |
| v2_full_v2_mixed_fg3 | v2_train_fs+extra (frame_gap=3, sw=3) | 307 | 5/10 | 0.18 | 0.28 | 0.64 | 0.89 | 0.985 | 0.30 | 0.56 |
| v2_full_v2_mixed_sw5 | v2_train_fs+extra | 307 | 4/10 | 0.04 | 0.43 | 0.61 | 0.67 | 0.982 | 0.36 | 0.40 |
| v2_full_v2_mixed_sw20 | v2_train_fs+extra | 307 | 4/10 | 0.05 | 0.42 | 0.64 | 0.78 | 0.961 | -- | 0.46 |
| v2_full_v2_mixed_drop0 (sw=10) | v2_train_fs+extra | 307 | 4/10 | 0.05 | 0.34 | 0.60 | 0.67 | 0.983 | -- | 0.48 |
| v2_full_v2_mixed_n12 (sw=10) | v2_train_fs+extra (n_obs=12) | 307 | 3/10 | 0.00 | 0.44 | 0.64 | 0.89 | 0.962 | -- | 0.36 |
| v2_full_v2_mixed_plw03 (sw=10) | v2_train_fs+extra (progress_w=0.3) | 307 | 3/10 | 0.02 | 0.42 | 0.64 | 0.78 | 0.972 | -- | 0.59 |
| v2_full_v2_mixed_plw5 (sw=10) | v2_train_fs+extra (progress_w=5) | 307 | 2/10 | 0.00 | 0.19 | 0.56 | **0.22** | **0.995** | -- | 0.24 |
| v2_full_v2_mixed_invfreq (sw=10) | v2_train_fs+extra (inverse-freq stage_w) | 307 | 3/10 | 0.08 | 0.38 | 0.64 | 0.67 | 0.974 | -- | 0.64 |
| v2_full_v2_mixed_24k (sw=10, 24k steps) | v2_train_fs+extra | 307 | 3/10 | 0.03 | 0.36 | 0.63 | 0.89 | 0.973 | -- | 0.72 |
| v2_full_v2_mixed_plw2 (sw=10, plw=2) | v2_train_fs+extra | 307 | 3/10 | 0.00 | 0.36 | 0.62 | **0.444** | 0.987 | -- | 0.60 |
| v2_full_v2_mixed_plw3 (sw=10, plw=3) | v2_train_fs+extra | 307 | 3/10 | 0.01 | 0.41 | 0.63 | 0.78 | 0.980 | -- | 0.49 |
| v2_full_v2_mixed_rewind5 (sw=10) | v2_train_fs+extra (max_rewind=5) | 307 | 3/10 | 0.02 | 0.41 | 0.54 | 0.44 | 0.990 | -- | 0.34 |
| **v2_full_v2_mixed_sw10 ckpt 7k** (early-stop) | v2_train_fs+extra | 307 | **3/10** | 0.04 | 0.29 | 0.57 | **0.111** | 0.995 | -- | 0.64 |
| v2_full_v2_mixed_plw15 (sw=10, plw=1.5) | v2_train_fs+extra | 307 | 2/10 | 0.01 | 0.22 | 0.57 | 0.33 | 0.994 | -- | 0.15 |
| v2_full_v2_mixed_epstart (sw=10, M4) | v2_train_fs+extra | 307 | 3/10 | 0.00 | 0.34 | 0.58 | 0.89 | 0.956 | -- | 0.59 |
| **v2_full_v2_mixed_paperfull (M2+M3+M4 sw=10) NEW BEST** | v2_train_fs+extra | 307 | **4/10** | 0.28 | 0.34 | 0.72 | **0.000** | **0.982** | -- | 0.67 |

## Open levers (to iterate, ordered)
1. Strict success-only training (ALL partials excluded, even labeled ones)
2. stage_loss_weight sweep: 1, 3 (current), 5, 10
3. frame_gap: 5 (current), 3, 10
4. drop_n_last_frames: 0, 1 (current), 3
5. success_threshold tuning at relabel time
6. CLIP backbone: B/32 (current), B/16, L/14
7. Inverse-frequency stage weights
8. Curriculum: pretrain succ-only → finetune w/ partials
9. annotation_mode: dual (current), sparse-only, dense-only
10. Larger n_obs_steps (current 8): 12, 16

## Brainstorm queue (when above exhausted)
- TBD
