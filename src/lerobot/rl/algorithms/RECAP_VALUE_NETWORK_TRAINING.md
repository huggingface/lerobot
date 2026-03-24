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
- A CSV with **fixed schema** containing per-episode success/failure labels.

### 1.1 Episode labels (bundled in the dataset)

Episode labels are stored inside the dataset at `meta/episode_labels.csv`. When you
load the dataset via `LeRobotDataset(repo_id=...)`, the labels file is downloaded
automatically alongside the other metadata.

The training script discovers the labels at `<dataset_root>/meta/episode_labels.csv`
by default — no extra flags needed. You can still override the path with
`--labels_csv_path` for local experimentation.

Example dataset: `jackvial/so101_pickplace_recap_merged_v2`

### 1.2 Required CSV format

```csv
episode_index,success
0,1
1,1
2,1
...
10,0
11,0
...
```

Rules:

- `episode_index` must match LeRobot episode indices.
- `success` must be binary (`1` success, `0` failure).
- Include a row for every episode used for training/validation.

If labels are missing for selected episodes, the script will raise an error.

### 1.3 Uploading labels to your own dataset

If you have your own LeRobot dataset on HuggingFace and want to bundle episode
labels so other users get them automatically:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="path/to/your/episode_labels.csv",
    path_in_repo="meta/episode_labels.csv",
    repo_id="your-username/your-dataset",
    repo_type="dataset",
)
```

### 1.4 Expected reward/return construction

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
  --repo_id="jackvial/so101_pickplace_recap_merged_v2" \
  --root="${HOME}/.cache/huggingface/lerobot" \
  --output_dir="${HOME}/code/lerobot/outputs/so101_pickplace_recap_merged_v2_value"
```

### 2.2 Quick start (pi0.5 backbone) (RTX 4070 TI SUPER)

```bash
uv run python -m lerobot.rl.algorithms.RECAPTrainValueNetwork \
  --repo_id="jackvial/so101_pickplace_recap_merged_v2" \
  --root="${HOME}/.cache/huggingface/lerobot" \
  --output_dir="${HOME}/code/lerobot/outputs/so101_pickplace_recap_merged_v2_value_4" \
  --pretrained_path="lerobot/pi05_base" \
  --epochs=2 \
  --batch_size=2 \
  --learning_rate=3e-4 \
  --num_workers=4 \
  --val_split_ratio=0.1 \
  --log_every_n_steps=10 \
  --validate_every_n_train_steps=50 \
  --plot_every_n_train_steps=200 \
  --max_val_steps_per_step_validation=20 \
  --c_fail=500.0 \
  --num_value_bins=16 \
  --val_plot_num_episodes=4 \
  --val_plot_num_frames=8 \
  --val_plot_every_n_epochs=1 \
  --model_precision="bfloat16" \
  --freeze_backbone=true
```

SmolVLA backbone (from scratch, full model)
```
uv run python -m lerobot.rl.algorithms.RECAPTrainSmolVLANetwork \
  --repo_id="jackvial/so101_pickplace_recap_merged_v2" \
  --root="${HOME}/.cache/huggingface/lerobot" \
  --output_dir="${HOME}/code/lerobot/outputs/so101_pickplace_recap_smolvla_scratch_0" \
  --epochs=20 \
  --batch_size=6 \
  --learning_rate=3e-3 \
  --num_workers=4 \
  --val_split_ratio=0.1 \
  --log_every_n_steps=20 \
  --validate_every_n_train_steps=50 \
  --plot_every_n_train_steps=200 \
  --max_val_steps_per_step_validation=20 \
  --c_fail=500.0 \
  --num_value_bins=56 \
  --val_plot_num_episodes=4 \
  --val_plot_num_frames=8 \
  --val_plot_every_n_epochs=1 \
  --load_vlm_weights=false \
  --freeze_vision_encoder=false \
  --freeze_backbone=false \
  --model_precision="bfloat16"
  ```

### 2.3 Full smoke-test command (step-based val + plots)

```bash
uv run python -m lerobot.rl.algorithms.RECAPTrainValueNetwork \
  --repo_id="jackvial/so101_pickplace_recap_merged_v2" \
  --root="${HOME}/.cache/huggingface/lerobot" \
  --output_dir="${HOME}/code/lerobot/outputs/so101_pickplace_recap_merged_v2_value_smoketest_1" \
  --epochs=1 \
  --batch_size=2 \
  --learning_rate=3e-4 \
  --num_workers=4 \
  --val_split_ratio=0.1 \
  --max_train_steps_per_epoch=200 \
  --max_val_steps_per_epoch=50 \
  --log_every_n_steps=10 \
  --validate_every_n_train_steps=25 \
  --plot_every_n_train_steps=100 \
  --max_val_steps_per_step_validation=10 \
  --c_fail=500.0 \
  --num_value_bins=16 \
  --val_plot_num_episodes=2 \
  --val_plot_num_frames=8 \
  --val_plot_every_n_epochs=1 \
  --paligemma_variant="gemma_300m" \
  --model_precision="bfloat16" \
  --freeze_vision_encoder=true
```

### 2.4 Initialising from pretrained pi0.5 weights

By default the backbone is randomly initialised, which means early training learns
only the unconditional distribution over value bins (the reconstructed return curve
will be flat). Initialising from pretrained pi0.5 VLM weights gives the model
meaningful visual features from step 0:

```bash
uv run python -m lerobot.rl.algorithms.RECAPTrainValueNetwork \
  --repo_id="jackvial/so101_pickplace_recap_merged_v2" \
  --root="${HOME}/.cache/huggingface/lerobot" \
  --output_dir="${HOME}/code/lerobot/outputs/so101_pickplace_recap_merged_v2_value_pretrained_1" \
  --pretrained_path="lerobot/pi05_base" \
  --epochs=2 \
  --batch_size=2 \
  --learning_rate=3e-4 \
  --num_workers=4 \
  --val_split_ratio=0.1 \
  --log_every_n_steps=10 \
  --validate_every_n_train_steps=50 \
  --plot_every_n_train_steps=200 \
  --max_val_steps_per_step_validation=20 \
  --c_fail=500.0 \
  --num_value_bins=16 \
  --val_plot_num_episodes=4 \
  --val_plot_num_frames=8 \
  --val_plot_every_n_epochs=1 \
  --model_precision="bfloat16" \
  --freeze_backbone=true
```

Notes:

- `--pretrained_path` accepts any HF Hub id or local path containing a pi0.5
  `model.safetensors` (e.g. `lerobot/pi05_base`).
- When set, the script **automatically overrides** `paligemma_variant` to `gemma_2b`
  (the VLM variant used by pi0.5 base) and sets `projection_dim=2048`.
- The vision tower, multi-modal projector, and language model weights are loaded;
  the fusion head and value head remain randomly initialised.
- **`--freeze_backbone=true` is strongly recommended** when using pretrained weights.
  It freezes the entire PaliGemma backbone (vision + language + projector) so only
  the fusion head and value head are trained. This preserves pretrained features,
  avoids OOM on 16GB GPUs, and focuses gradient signal on the value prediction heads.
- `--freeze_vision_encoder=true` freezes only the vision tower (the language model
  and projector remain trainable). This needs more VRAM but may improve results if
  the domain is very different from pi0.5's pretraining data.
- The full backbone is ~2B parameters (vs ~300M with `gemma_300m`). With
  `--freeze_backbone=true` only ~12M parameters are trained (fusion + value heads).

### 2.5 What the script trains

- Base backbone: lighter PI0/PI05-style PaliGemma language+vision stack (no action expert head).
- No subtask labeling/tokenization path is used in this value-network trainer.
- Head: distributional value head over bins.
- Loss: cross-entropy on return bins.

No TD bootstrap target is used in this training path.

For short-horizon tasks, this lighter setup is usually sufficient for value learning. The richer
`pi05_full` stack (with subtask annotation pathways and additional conditioning) tends to matter
more for full policy/action modeling than for this standalone value head.

Reference full-network implementation:

- [cijerezg/lerobot `pi05_full` (my-pi05-merge)](https://github.com/cijerezg/lerobot/tree/my-pi05-merge/src/lerobot/policies/pi05_full)

---

## 3) Outputs And What To Expect

Outputs are written under `--output_dir`:

- `train_config.json`
- `metrics_history.json`
- `checkpoints/last.pt`
- `checkpoints/best.pt`
- `validation_plots/epoch_XXX/episode_YYYYY.png` (when plotting is enabled)

### 3.1 Metrics

Each epoch logs:

- `train_loss`, `val_loss`: CE over value bins
- `train_bin_acc`, `val_bin_acc`: top-1 bin accuracy
- `train_value_mae`, `val_value_mae`: MAE between expected value and continuous target

Fine-grained progress logs can be enabled/tuned with:

- `--log_every_n_steps`: print train/val running metrics, throughput, and ETA every N steps
- `--validate_every_n_train_steps`: run an extra validation pass every N train steps (`0` disables)
- `--max_val_steps_per_step_validation`: cap validation steps for those extra step-based passes

### 3.2 Validation trajectory plots

The trainer can save paper-style per-episode validation visualizations:

- top strip: a few sampled frames from the trajectory
- title: episode id + `SUCCESS`/`FAIL` label
- return panel:
  - labeled expected return from supervision (`target_value`)
  - reconstructed expected return from predicted bin distribution:
    `E[v] = sum_b p(b|x_t) * v_b`

Plot controls:

- `--val_plot_num_episodes`: how many validation episodes to visualize per plotted epoch (set `0` to disable)
- `--val_plot_num_frames`: how many preview frames to show in the top strip
- `--val_plot_every_n_epochs`: save plots every N epochs
- `--plot_every_n_train_steps`: save validation plots every N train steps (`0` disables)
- plotted episodes are collected with a dedicated full-trajectory pass, so plots cover complete episode trajectories even if validation metrics are step-capped

### 3.3 Typical behavior

- Early epochs: rapid drop in CE loss.
- Bin accuracy usually improves before MAE fully stabilizes.
- Validation curves are the main signal; use `best.pt` for downstream usage.

### 3.4 Frequent issues

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
  - `src/lerobot/policies/pi05/processor_pi05.py`
- Actor/critic update orchestration:
  - `src/lerobot/rl/pi05_train_utils.py`
- Current RL policy/critic plumbing:
  - `src/lerobot/rl/rl_pi05.py`

If you need the full-network variant with subtask annotations, see:

- [cijerezg/lerobot `pi05_full` (my-pi05-merge)](https://github.com/cijerezg/lerobot/tree/my-pi05-merge/src/lerobot/policies/pi05_full)

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
