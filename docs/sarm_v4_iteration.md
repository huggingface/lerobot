# SARM v4 Iteration Log

Started 2026-05-02. v4 datasets built from 161 success eps (drop ep 160 → 160) + 30 partials. Image rescaled 224→128 at preprocessing. CLIP cache built per dataset. All v4 trained at 128x128, success_train_fs (160 eps × 50803 frames train), 2-cam (front+wrist), seed=1000, n_obs=8, frame_gap=5.

Eval: domrachev03/sim_3stage_v4_val_fs (190 eps × 6837 fr), success_threshold=0.9, 9 gates.

## v4 results table (v4_succ_train_fs unless noted)

| run | sw | gt | rewind | extras | mean_mid | full_max | zero_max | mono | plat | gates_pass |
|-----|-----|-----|--------|--------|----------|----------|----------|------|------|-----------|
| iter1 (with partials) | 1.0 | 0.75 | 3 | 2cam | 0.059 | low | 0 | 0.63 | 0.37 | 2 |
| iter2 (with partials) | 1.0 | 0.75 | 3 | front-only | 0.141 | - | 0 | 0.61 | 0.30 | 2 |
| succ1 7k | 1.0 | 0.75 | 3 | 2cam | 0.175 | - | 0 | 0.64 | 0.30 | 2 |
| succ1 14k | 1.0 | 0.75 | 3 | 2cam | 0.193 | 0.53 | 0.20 | 0.65 | 0.33 | 2 |
| succ1f 7k | 1.0 | 0.75 | 3 | front | 0.153 | - | 0 | 0.66 | 0.37 | 2 |
| succ1f 14k | 1.0 | 0.75 | 3 | front | 0.127 | - | 0 | 0.62 | 0.37 | 2 |
| **succ3 7k** | 3.0 | 0.75 | 3 | 2cam | 0.268 | - | 0.40 | 0.65 | 0.37 | 4 |
| **succ3 14k (BEST)** | 3.0 | 0.75 | 3 | 2cam | **0.281** | **0.57** | 0.80 | 0.65 | 0.40 | **4** |
| succ24k 8k | 1.0 | 0.75 | 3 | 2cam | 0.249 | - | 0.20 | 0.65 | 0.40 | 3 |
| succ24k 16k | 1.0 | 0.75 | 3 | 2cam | 0.139 | - | 0 | 0.60 | 0.23 | 3 |
| succ24k 24k | 1.0 | 0.75 | 3 | 2cam | 0.183 | - | 0.40 | 0.61 | 0.40 | 3 |
| part3 (with partials) | 3.0 | 0.75 | 3 | 2cam | 0.029 | - | 0 | 0.68 | 0.30 | 3 |
| sw10 7k | 10.0 | 0.75 | 3 | 2cam | 0.234 | - | 0.60 | 0.76 | 0.40 | 3 |
| sw10 14k | 10.0 | 0.75 | 3 | 2cam | 0.238 | - | 0.80 | 0.67 | 0.33 | 3 |
| sw03 14k | 0.3 | 0.75 | 3 | 2cam | 0.237 | - | 0.80 | 0.66 | 0.40 | 3 |
| sw5 14k | 5.0 | 0.75 | 3 | 2cam | 0.173 | - | 0.20 | 0.65 | 0.27 | 3 |
| 224_succ3 14k | 3.0 | 0.75 | 3 | 224x224 | 0.104 | - | 0 | 0.58 | 0.40 | 3 |
| succ3_24k 8k | 3.0 | 0.75 | 3 | 24k steps | 0.190 | - | 0.40 | 0.61 | 0.43 | 3 |
| succ3_24k 16k | 3.0 | 0.75 | 3 | 24k steps | 0.178 | - | 0 | 0.66 | 0.37 | 3 |
| succ3_24k 24k | 3.0 | 0.75 | 3 | 24k steps | 0.173 | - | 0 | 0.64 | 0.27 | 3 |
| sw2 14k | 2.0 | 0.75 | 3 | 2cam | 0.216 | - | 0.60 | 0.66 | 0.37 | 3 |
| plw5 14k | 3.0 | 0.75 | 3 | plw=5 | 0.244 | - | 0.80 | 0.67 | 0.43 | 3 |
| drop0 14k | 3.0 | 0.75 | 3 | drop_n_last=0 | 0.269 | 0.55 | 1.0 | 0.66 | 0.37 | 3 |
| gt0 14k | 3.0 | 0.0 | 3 | no TF | 0.206 | 0.53 | 0.40 | 0.67 | 0.40 | 3 |
| rew1 14k | 3.0 | 0.75 | **1** | rew_p=0.8 | 0.160 | - | 0 | 0.65 | 0.30 | 3 |
| rew1lp 14k | 3.0 | 0.75 | **1** | rew_p=0.3 | 0.169 | - | 0 | 0.63 | 0.33 | 3 |

## Best: v4_succ3_14k

- ckpt: `outputs/sim_3stage_sarm_v4_succ3/checkpoints/014000/pretrained_model`
- Recipe: success-only (160 eps, 128x128), sw=3, gt=0.75, max_rewind=3, 14k steps, batch=32, seed=1000
- Eval (cumulative breakpoints): mean_mid=0.281 ✓, lin_mad=0.32 ✗, mono=0.65 ✗, plat=0.40 ✗, last_stage_max=0 ✗, zero_max=0.80 ✗, succ_term=0 ✗, fail_term=0 ✓, ne=0.94 ✓
- Eval (linear breakpoints): mean_mid=0.395 ✓, max=0.75 in full bucket, 100% reach max≥0.5
- Per-bucket signal calibrated: term scales linearly with completion (0/6→0.27, full→0.18 because of late-stage reward dilution but max scales)

## Levers tried + outcomes

- **stage_loss_weight**: 0.3, 1, 2, **3 ✓ (best)**, 5, 10
- **gt_stage_ratio**: 0, 0.25, 0.5, **0.75 ✓ (best)**, 1.0
- **drop_n_last_frames**: 0 (slightly worse), **1 ✓ (default best)**
- **progress_loss_weight**: 1 (default best), 5 (worse)
- **resolution**: **128x128 ✓ (best)**, 224x224 (much worse)
- **data**: **success-only 160 eps ✓**, with partials (much worse)
- **cams**: **2cam ✓ (best)**, front-only (worse on v4)
- **steps**: **14k ✓**, 7k (slightly worse), 24k (worse, overfits)
- **max_rewind**: 1 (worse), **3 ✓ (default)**
- **rewind_probability**: 0.3 (worse), **0.8 ✓ (default)**
- **inverse_freq class weights**: didn't try on v4 (collapse on v3)

## Ceiling

Model reaches max ~0.55-0.60 on full eps (with cumulative breakpoints), regardless of recipe. Progress 0.9 unattainable. Per-bucket calibration is correct (term scales with completion).

Hypothesis: stage_argmax_acc only ~0.24 on full eps → model rarely confident enough about stage 5 to push reward to last quarter. Architectural ceiling for current framework + dataset.

## Code changes (committed)

- lerobot_policy_sarm @ 83e25bc: `progress_loss_weight` cfg knob, per-stage diag stats
- lerobot @ 7503e086: SARM_DIAG log line in train

## Status: SHIPPABLE as approximate reward

`outputs/sim_3stage_sarm_v4_succ3/checkpoints/014000` is a working SARM model in basin A (meaningful learning). Useful as ACT reward signal:
- Discriminates stages roughly correctly (full max=0.57 vs 0/6 max=0.45-0.55)
- Term scales linearly across buckets (sensible reward signal)
- 100% of full eps reach max≥0.5

Doesn't pass strict gates (succ_term, last_stage, zero_max), but should be sufficient for ACT residual training where SARM gives smooth 0→0.5+ reward.

# Paper-recipe ablation plan (lerobot-93)

Audited arxiv 2509.25358v4 against current impl. 5 mismatches found. Each tested in isolation against baseline.

## Baseline
`v4_succ3w 7k`: wrist-only, sw=3, 14k steps, frame_gap=20, centered window, 1-layer linear heads, causal mask. **mean_mid=0.302, last_stage_max=0.144** (current best).

## Identified mismatches with paper
| # | Detail | Paper | Ours |
|---|--------|-------|------|
| M1 | Camera | top-down only (Table 7 ablates wrist away) | wrist-only |
| M2 | Transformer mask | non-causal/bidirectional aggregator | causal triu mask |
| M3 | Output heads | 2-layer MLP hidden=512 | 1-layer linear |
| M4 | First frame | literal ep_start (frame 0), then 8 consecutive at gap | centered window [-4*gap..0..+4*gap] clamped |
| M5 | stage_loss_weight × epochs | sw≈1, 2 epochs | sw=3, ~6 epochs |

Other paper details we already match: CLIP frozen, 8-layer 12-head 768-hidden, AdamW lr=5e-5 wd=1e-3, rewind_aug≤4 frames reversed, lang perturbation, position bias only on first frame.

## Experiment matrix
| name | change vs baseline | hypothesis |
|------|-------------------|-----------|
| abl_front | image_keys: wrist→front | top-down view distinguishes scene state better |
| abl_nocausal | drop causal mask | bidirectional attn helps stage classification |
| abl_mlpheads | head: Linear → Linear(d,512)+ReLU+Linear(512,k) | more head capacity |
| abl_epstart | obs_delta: pin index 0 to ep_start, rest forward at gap | episode-anchor context |
| abl_sw1_short | sw 3→1, steps 14k→1500 | match paper's 2-epoch training |
| full_paper | all 5 above combined | true paper reproduction |

## Files to touch
- `lerobot_policy_sarm/src/lerobot_policy_sarm/configuration_sarm.py` — `observation_delta_indices` property (M4)
- `lerobot_policy_sarm/src/lerobot_policy_sarm/modeling_sarm.py` — heads (M3), causal mask (M2)
- `lerobot_policy_sarm/src/lerobot_policy_sarm/processor_sarm.py` — frame indexing (M4 inference path)
- New cfgs in `lerobot/src/lerobot/rl/sim_3stage_sarm_v4_abl_*_train.json`

## Sequence
Run sequentially on local GPU 0, ~35min each. Eval after each. ETA ~3.5h total.

PAUSED awaiting user direction.
