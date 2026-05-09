# ACT iteration log (lerobot-123)

Eval: gym_manipulator + SARM-reward env, n=20 rollouts. Use ACT ckpt's training stats.

## Criteria (must pass all)
- success_threshold = 0.95 cumulative SARM progress
- ≥10% rollouts hit success
- ≥80% rollouts reach last stage (max_cum_progress > end-of-stage-5 boundary ≈ 0.633 for v2 sparse_temporal_proportions)

## Resource
- Native sim renderer = 128 (env.py modified)
- Eval on same GPU as training

## Results (RA-BC, all evaluated w/ SARM-reward env @ native 128)
| name | data | SARM-relabel | mean_term | max_term | mean_max_cum | reach_st6 | succ@0.95 |
|---|---|---|---|---|---|---|---|
| v2_extra_rabc_v11 chunk10 80k | 60s v2_extra | v2_nostale_2cam | 0.42 | 0.75 | - | 40% | 0% |
| v4_rabc_v11 80k | 160 v4 succ | v2_nostale_2cam | 0.53 | 0.76 | - | 25% | 0% |
| kappa005 40k (peak) | 60 v2_extra | v2_nostale_2cam | 0.61 | 0.90 | - | 40% | 0% |
| chunk20_v2_extra 80k | 60 v2_extra | v2_nostale_2cam | 0.57 | 0.88 (native 128) | - | - | - |
| chunk20_v2_full 80k | 288 v2_full | v2_nostale_2cam | 0.64 | 0.78 | 0.94 | 80%* | 0% |
| long 40k (peak) | 60 v2_extra | v2_nostale_2cam | 0.49 | 0.94 | - | - | 0% |
| **chunk20 v2_full_v2_succonly 80k BASELINE** | 205 succ | v2_full_v2_succonly_sw3 | 0.62 | 0.82 | 0.78 (range 0.75-0.85) | **100%** | 0% |

*Caveat: v2_full reach_st6 high but visually bad. SARM hallucinations on rollouts.

## Plan
1. **Baseline (Phase 1, regardless of SARM gates)**: ACT chunk20 RA-BC trained on v2_full_v2_succonly_SARM-relabel
2. **Production**: ACT chunk20 RA-BC trained on optimal-SARM-relabel
3. Iterate knobs only if criteria not met:
   - kappa: 0.005, 0.01, 0.02
   - chunk_size: 10, 20
   - augmentation: max_num_transforms 3 vs 5
   - state_noise_std: 0, 0.01
   - longer training (160k)

## Brainstorm queue (when above exhausted)
- TBD
