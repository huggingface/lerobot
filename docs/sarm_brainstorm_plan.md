# SARM v4 fallback brainstorming plan

Triggers: if state-noise + short-training + invfreq + epstart all fail to beat baseline 0.302.

## Phase 1: parallel deep-analysis agents (run concurrently)

**Agent A — code audit**
- Read `lerobot_policy_sarm/src/lerobot_policy_sarm/{configuration_sarm.py,modeling_sarm.py,processor_sarm.py,sarm_utils.py}` end-to-end.
- Compare to paper appendix A.4 line-by-line for:
  - Loss formulation (CE coeffs, MSE coeffs, regularization)
  - Stage embedding wiring (one-hot vs learned, where injected)
  - Optimizer schedule (cosine warmup vs paper's constant)
  - Causal mask, position bias, language perturbation pool source
  - Numerical stability (logits clamp, softmax temperature, log-spaces)
- Flag any silent bugs, wrong loss weighting, off-by-one in indexing, dtype mismatches.

**Agent B — dataset audit**
- Read `simulator_for_IL_RL/simulator_for_il_rl/env.py` — how demos are recorded.
- Read `lerobot/scripts_local/build_3stage_v4_*.py`, `frame_stride_split_*.py`, `write_temporal_proportions.py` — how dataset is built.
- Check: stage-boundary determination (script vs operator-pressed), label noise sources, frame indexing edge cases (off-by-1 at ep_start/ep_end), proprio normalization (mean/std), CLIP cache integrity (does cache match data?), any silent failures in merge_datasets, episode drop logic.
- Compare v4 vs v3 vs v2 build pipelines for any quiet drift.

**Agent C — experiment synthesis**
- Read `lerobot/docs/sarm_v4_iteration.md` + this brainstorm doc + recent eval summaries.
- Catalog: every recipe tried, every metric, every "this hurt / this helped" result.
- Find unexplored axes by exclusion. What category has zero data points? Examples to consider: optimizer betas, LR schedule shape (linear vs cosine vs OneCycle), batch size sweep, dropout rate, label smoothing, mixup, weight decay, warmup length, gradient clipping, EMA, model size scaling (smaller as well as bigger).
- Surface 5-10 candidate experiments not covered by current ablation tree.

**Agent D — paper deep dive**
- Re-read `2509.25358v4.pdf` focusing on:
  - Sec 4 Method (optimization details)
  - Sec 5/6 Results (subtle tricks that drove their numbers)
  - Appendix A.4 (full hyperparameters)
  - Appendix B/C (any task-specific recipes)
  - Reference to ReWiND (their primary baseline) — what did ReWiND do differently?
- Flag any detail that the team would consider "obvious" but is implicit.

## Phase 2: synthesis (single agent or me)

- Aggregate findings; cross-reference across agents to find consensus signals.
- Rank candidate fixes by (impact × cost) and (likelihood × leverage).
- Write a prioritized list of 5-10 next experiments.

## Phase 3: implementation

- Implement top 3 experiments simultaneously (if independent).
- Use server (4 GPUs) and local in parallel.
- Eval on v3_eval (held-out) + full-traj for honest signal.
- Track in beads.

## Categories to explore (if all current ideas fail)

1. **Architecture**: shared vs split stage/subtask transformer, cross-attention between heads, smaller model (paper says 60M optimal, ours is 120M), bigger MLP fusion, ResNet visual backbone vs CLIP.
2. **Loss**: label smoothing on stage CE, focal loss, contrastive between adjacent stages, KL divergence between predicted and "calibrated" stage distributions.
3. **Training**: 2 epochs only (paper), different LR schedule, EMA, gradient accumulation, larger batch, mixed precision toggle.
4. **Data pipeline**: episode-level holdout split (true generalization), stratified sampling per stage, aggressive image aug (color jitter, blur, crop), mixup.
5. **Inference**: ensemble across multiple ckpts, calibration via temperature scaling, post-hoc smoothing of stage predictions.
6. **Postprocessing of dataset**: re-annotate stage boundaries with VLM, smooth noisy boundaries, drop suspect annotations.

## Action

Wait for current 4 (short3500 / noise005 / noise01 / noise005_short) eval results.
If max mean_mid < 0.302 across all of them, trigger Phase 1 (4 parallel agents).
