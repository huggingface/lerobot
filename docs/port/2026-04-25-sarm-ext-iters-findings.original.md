# SARM-ext iter findings (2026-04-25 cycle)

Epic lerobot-33.

## Gate key
- succ_term ≥ 0.9 at ≥95% on full eps
- fail_term ≥ 0.9 at 0% on partial eps
- 0/4 max ≥ 0.5 at 0%
- lin_mad ≤ 0.25, mean_mid ≥ 0.25, monotonicity ≥ 0.85
- **stage_not_exceed ≥ 0.9** (T5 new: pred_stage ≤ gt_cur_stage per frame)
- plateau_ok at ±0.10 of breakpoint on partial buckets

## Iter table

| iter | dataset | cams | knobs | succ | fail | 0/4max | lin_mad | mean_mid | mono | stage_ne | plateau | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iter4 (champ) | merged (103ep; leak src1) | wrist | sarm, dual, n8 g5 r3 slw3 | 0.975✅ | 0.00✅ | 0.67❌ | 0.106✅ | 0.65✅ | 0.94✅ | 0.89❌ | 0.48❌ | FAIL 0/4+plateau+stg_ne |
| iter5 | merged | front+wrist | sarm_ext, dual, n8 g5 r3 slw3 | | | | | | | | | TRAINING |

## Log

### iter-4 rescore on merged (103 eps)
- Used champion iter-4 ckpt (type=sarm upstream, wrist-only).
- Merged ds includes _filtered (leaky: overlaps iter-4 training) + _2 (unseen).
- **Per-bucket**:
  - 0/4 (n=18): term=0.24 max=0.59; **67% spike≥0.5** → fail
  - 1/4 (n=15): term=0.20 max=0.43; plateau 40%
  - 2/4 (n=15): term=0.29 max=0.64; plateau 47%
  - 3/4 (n=15): term=0.62 max=0.79; plateau 87%
  - full (n=40): term=0.99 max=0.99 succ=97%
- **Root cause**: wrist-only loses scene context → background motion triggers false progress on 0/4. Partial plateau drift reflects stage confusion past last-completed breakpoint.
- **Fix**: 2-cam front+wrist (iter-5).

### iter-5 (plan)
- Ext SARM, 2-cam (front+wrist), merged ds, dual mode. 5000 steps batch16.
- Expected: stage_ne up (2-cam adds context), lin_mad steady, maybe mean_mid down w/ partials.

## Brainstorm backlog (T6 knob sweep if iter-5 insufficient)
- `stage_loss_weight`: 3→5→10
- `rewind_probability`: 0.8 default → lower? Might reduce noise on partials
- `drop_n_last_frames`: 1 → 3
- `n_obs_steps`: 8 → 12
- `frame_gap`: 5 → 3 (tighter window, more frames/sec)
- Force features (proprio + wrench) as extra state channel
- τ-cap: clamp tau pred to [0, breakpoint_of_stage]
- Cross-attn fusion (front↔wrist) instead of concat
- Augmentation: color jitter, random crop (train time only)
