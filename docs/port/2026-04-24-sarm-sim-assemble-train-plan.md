# SARM sim_assemble train plan (caveman)

Goal: train SARM that detects 4-stage progress on `AssembleBase-v0` *and* does not false-positive on failures. No SARM code edits. Tweak: config + dataset prep only.

## Data (on hand)

- TRAIN: `domrachev03/sim_assemble_sarm_multistage_two_stages_filtered_v2` (50 eps, 10778 frames, 20 fps)
  - n_stages dist: {0:5, 1:5, 2:5, 3:5, 4:30}
  - partial eps carry only the *completed* stages (last-entered partial stage dropped at record time → progress frozen at breakpoint of last completed stage)
  - stages: `approach_box, bring_box, approach_target, place_target_in_the_box`
  - tp_sparse: ~0.24 / 0.27 / 0.24 / 0.26 (already computed from completed stages only; 0-stage eps contribute 0 frames)
- VAL-SUCC: `domrachev03/sim_assemble_demos_two_stages` (30 full-success eps, 5442 frames, 20 fps, NO subtask labels)
  - usable only for: "terminal ≥0.9 on success", "monotone rise", "mean peak"
  - not usable for per-stage GT metrics

## Panda lessons recap

1. train-inference parity critical (ring-buffer bug cost 60pp TP).
2. stage-loss-weight ≥3 needed or stage head collapses to argmax=0.
3. operator stage labels > heuristic splits (for linearity).
4. τ-cap / partial-drop is the trick against step-function collapse. **We already partial-dropped → expect clean separation without τ-cap hacks.**
5. aug 3x (color+crop) + stage-weight 10 won OOD.
6. task-text mismatch → SARM silently → 0 everywhere. Pin `task` string.
7. normalizer stats from empty `-train` split → OOD collapse. Always pin `stats_dataset_repo_id` to full set.

## Panda gates (must meet)

- Success terminal≥0.9: ≥95%
- Failure terminal≥0.9: 0%
- Failure max≥0.5: 0% (soft: max≥0.25 OK for multi-stage since breakpoints aren't all <0.25)
- lin_mad (per-ep, on full-success): ≤0.25
- mean_mid (on full-success): ≥0.25

## Our added gates (4-stage specific)

- per-stage plateau accuracy: k/4 partial ep → `peak progress ∈ [bp[k] - 0.10, bp[k] + 0.10]` (±0.10 tolerance)
  - e.g. 2/4 ep (bp[2]≈0.50): peak in [0.40, 0.60]
  - 0/4 ep (bp[0]=0): peak ≤ 0.10 (strict FP guard)
  - 3/4 ep (bp[3]≈0.74): peak in [0.64, 0.84]
- stage-argmax accuracy (on GT-labeled frames, full-success val): ≥80% frame-wise
- monotonicity rate (per-ep, success): ≥0.85 of frames have progress[t] ≥ progress[t-1] (after mild smoothing)
- transition-lag (detected stage change wrt GT boundary): within ±15% of stage length
- **no false-positive stage advance on 0/4 eps**: stage argmax == 0 for ≥90% of frames across 0/4 eps

## Preprocessing plan

1. **split** filtered_v2 into train/val:
   - stratified holdout: take ep indices where `ep_idx % 5 == 0` from each stage-count bucket:
     - 4/4 bucket (30 eps) → 6 val, 24 train
     - 3/4 (5) → 1 val, 4 train
     - 2/4 (5) → 1 val, 4 train
     - 1/4 (5) → 1 val, 4 train
     - 0/4 (5) → 1 val, 4 train
   - total: 10 val / 40 train
   - rebuild *two* new LeRobotDataset copies (`..._train`, `..._val`) rather than frame-level split (panda did frame-level; for SARM-per-ep eval we need ep-level cleanliness)
2. **stats**: both splits inherit stats from the full filtered_v2 (stats_dataset_repo_id pin)
3. **temporal_proportions**: copy from full filtered_v2 verbatim. Do NOT recompute from the train split alone (too small, biased).
4. **demos val**: leave `sim_assemble_demos_two_stages` unlabeled — use only as success-only external val set.

## Training plan

### Baseline A — success-only (naive panda iter-0 analogue)
- dataset: sim_assemble_demos_two_stages (single_stage, no subtask labels)
- annotation_mode: single_stage
- expected failure mode: high FP on failure eps (they'll all look like "almost done")
- serves as null-baseline for FP metric comparison

### Baseline B — full dataset (main contender)
- dataset: filtered_v2_train (40 eps, 4-stage annotated + 0-stage Nones)
- annotation_mode: dual (panda-proven path for multi-stage)
- stage_loss_weight: 3.0 (panda iter 12 best)
- n_obs_steps: 4, frame_gap: 5 (window = 1.0s at 20fps, matches panda lift-box 1s window)
- max_rewind_steps: 2
- steps: 5000
- batch_size: 16
- image_key: observation.images.front (panda pattern)

### Iteration knobs (if gates miss)
- stage_loss_weight sweep: {1, 3, 10}
- frame_gap sweep: {2, 5, 10} (0.1s / 0.25s / 0.5s windows)
- n_obs_steps: {4, 8}
- rewind_probability: {0.5, 0.8}
- dropout: {0.1, 0.2}
- image_key dual: front vs wrist (wrist closer to object)
- NO code edits to SARM/modeling/processor.

## Eval protocol

1. For each ckpt:
   a. Run SARM offline on each val dataset, record per-frame (stage_probs, progress).
   b. Per-ep compute: terminal, max, mean_mid, lin_mad (vs linear ref), stage_argmax_accuracy, monotonicity.
   c. Aggregate by ep-type (full-success / k-of-4 partial / demos-success).
   d. Dump `metrics.json` + `summary.md` + per-ep `png` plots (progress curve + stage probs + GT boundaries overlay).
2. Compare to gates. PASS = all gates met across all buckets.

## Deliverables

- `src/lerobot/rl/sim_assembling_sarm_multistage_train.json` — train config (baseline B)
- `src/lerobot/rl/sim_assembling_sarm_single_stage_train.json` — train config (baseline A)
- `/tmp/sarm_split_sim_assemble.py` — stratified split
- `src/lerobot/policies/sarm/eval_sarm_sim_assemble.py` — eval harness (ported from panda `viz_sarm_progress.py`)
- `docs/port/2026-04-24-sarm-sim-assemble-findings.md` — running log of each iter: config, metrics table, verdict
- final ckpt under `outputs/sim_assemble_sarm_<iter>/checkpoints/last/pretrained_model`

## No-go / not-allowed

- no SARM model / processor / utils edits
- no push / commit (user verifies)
- no net edits to sim_assembling env

## Iter table (to fill)

| iter | cfg | train data | stage_loss_w | n_obs / gap | SuccTerm | FailTerm | FailMax0.5 | lin_mad | stage_argmax | plateau k-of-4 | verdict |
|-----|-----|-----------|-------------|-----------|---------|---------|-----------|---------|-------------|----------------|---------|
| 0 (A) | single_stage | demos (30) | n/a | 4/5 | ?       | ?       | ?         | ?       | n/a         | n/a            | ?       |
| 0 (B) | dual stage=3 | filtered_v2_train (40) | 3 | 4/5 | ?       | ?       | ?         | ?       | ?           | ?              | ?       |

(fill per-iter as we go)
