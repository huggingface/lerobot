# RECAP Value Network Training

This document explains how to:

1. Prepare a LeRobot dataset + success/fail labels for value training.
2. Train the standalone RECAP value network.
3. Interpret outputs and training behavior.
4. Connect the trained value network to the main advantage-conditioned policy training loop.

The training entrypoint is:

- `src/lerobot/rl/algorithms/RECAPTrainValueNetwork.py`

The model is:

- `src/lerobot/rl/algorithms/RECAPValueNetwork.py`

---

## 1) Dataset Preparation

The value trainer expects:

- A valid LeRobot dataset (with episode metadata and frames).
- A CSV with **fixed schema**:
  - `episode_index`
  - `success` (0 or 1)

### 1.1 Required CSV format

Example:

```csv
episode_index,success
0,1
1,0
2,1
3,1
```

Rules:

- `episode_index` must match LeRobot episode indices.
- `success` must be binary (`1` success, `0` failure).
- Include a row for every episode used for training/validation.

If labels are missing for selected episodes, the script will raise an error.

### 1.2 Expected reward/return construction

During preprocessing, the script builds paper-style targets from episode outcomes:

- Per-step reward:
  - non-terminal steps: `-1`
  - terminal success: `0`
  - terminal failure: `-C_fail` (configurable as `--c_fail`)
- Empirical return is computed via reverse cumulative sum.
- Return is normalized by per-task max episode length.
- Normalized values are clamped to `[-1, 0]`.
- Values are discretized into `B` bins (`--num_value_bins`, default `201`).

This produces frame-level supervision targets:

- continuous normalized return (`target_value`)
- discrete return bin (`target_bin`)

---

## 2) Training

Use `uv` from the repo root.

### 2.1 Minimal command

```bash
uv run python -m lerobot.rl.algorithms.RECAPTrainValueNetwork \
  --repo_id="your_org/your_dataset" \
  --root="/path/to/local/dataset/root" \
  --labels_csv_path="/path/to/episode_labels.csv" \
  --output_dir="/path/to/output/recap_value"
```

### 2.2 Common useful flags

```bash
uv run python -m lerobot.rl.algorithms.RECAPTrainValueNetwork \
  --repo_id="your_org/your_dataset" \
  --root="/path/to/local/dataset/root" \
  --labels_csv_path="/path/to/episode_labels.csv" \
  --output_dir="/path/to/output/recap_value" \
  --epochs=20 \
  --batch_size=8 \
  --learning_rate=3e-4 \
  --num_workers=4 \
  --val_split_ratio=0.1 \
  --c_fail=24.0 \
  --num_value_bins=201 \
  --paligemma_variant="gemma_300m" \
  --model_precision="float32" \
  --freeze_vision_encoder=false
```

### 2.3 What the script trains

- Base backbone: PI0.5/PI05-style PaliGemma language+vision stack (no action expert head).
- Head: distributional value head over bins.
- Loss: cross-entropy on return bins.

No TD bootstrap target is used in this training path.

---

## 3) Outputs And What To Expect

Outputs are written under `--output_dir`:

- `train_config.json`
- `metrics_history.json`
- `checkpoints/last.pt`
- `checkpoints/best.pt`

### 3.1 Metrics

Each epoch logs:

- `train_loss`, `val_loss`: CE over value bins
- `train_bin_acc`, `val_bin_acc`: top-1 bin accuracy
- `train_value_mae`, `val_value_mae`: MAE between expected value and continuous target

### 3.2 Typical behavior

- Early epochs: rapid drop in CE loss.
- Bin accuracy usually improves before MAE fully stabilizes.
- Validation curves are the main signal; use `best.pt` for downstream usage.

### 3.3 Frequent issues

- Missing labels: CSV does not cover all selected episodes.
- Bad mapping: CSV `episode_index` does not match dataset episodes.
- OOM: lower `--batch_size` or use smaller `--paligemma_variant`.

---

## 4) How This Ties Into Advantage-Conditioned Policy Training

The value network is trained first, then used to construct the advantage indicator consumed by the policy.

High-level flow:

1. Train `RECAPValueNetwork` on accumulated data (demonstrations + rollouts).
2. Run value inference to get scalar expected values.
3. Compute advantage proxy and threshold into binary indicator `I_t`.
4. Inject `I_t` as textual conditioning (`Advantage: positive|negative`) for actor policy training.

### 4.1 Practical integration points in this repo

- Advantage prompt formatting:
  - `src/lerobot/policies/pi05_full/processor_pi05.py`
- Actor/critic update orchestration:
  - `src/lerobot/rl/pi05_train_utils.py`
- Current RL policy/critic plumbing:
  - `src/lerobot/rl/rl_pi05.py`

### 4.2 Recommended integration logic

- Replace TD-style critic-value source used for indicator construction with outputs from `RECAPValueNetwork`.
- Compute a binary indicator per sample (task-dependent thresholding is recommended for paper alignment).
- Keep intervention override behavior (force positive indicator for interventions) if desired.

This keeps the policy training phase aligned with the RECAP principle:

- value model learns progress-to-success
- policy is optimized through advantage-conditioned supervision

---

## 5) Quick Checklist

- [ ] LeRobot dataset loads locally.
- [ ] CSV has `episode_index,success`.
- [ ] Labels cover all training episodes.
- [ ] Value training completes and `checkpoints/best.pt` is produced.
- [ ] Expected values are used to generate policy advantage indicators.
