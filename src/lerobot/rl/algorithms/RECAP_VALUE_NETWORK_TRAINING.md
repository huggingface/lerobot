# RECAP Value Network Training

This document explains how to:

1. Prepare a LeRobot dataset + success/fail labels for value training.
2. Train the standalone RECAP value network.
3. Interpret outputs and training behavior.
4. Connect the trained value network to the main advantage-conditioned policy training loop.

There are two backbone variants:

| Variant | Training entrypoint | Model |
|---------|-------------------|-------|
| **pi0.5 (PaliGemma)** | `src/lerobot/rl/algorithms/RECAPTrainValueNetwork.py` | `src/lerobot/rl/algorithms/RECAPValueNetwork.py` |
| **SmolVLA (SmolVLM2)** | `src/lerobot/rl/algorithms/RECAPTrainSmolVLANetwork.py` | `src/lerobot/rl/algorithms/RECAPSmolVLAValueNetwork.py` |

The SmolVLA variant replaces the PaliGemma vision-language backbone with
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. It is smaller, faster to train, and
does not require a pretrained checkpoint — weights can be loaded from HuggingFace
directly via `--load_vlm_weights=true` or trained from scratch with
`--load_vlm_weights=false`. Additional flags `--freeze_vision_encoder` and
`--freeze_backbone` give fine-grained control over which parts of the SmolVLM2
backbone are trainable.

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

### 2.1 Quick start — pi0.5 backbone (RTX 4070 TI SUPER)

Uses the PaliGemma backbone with a pretrained pi0.5 checkpoint. The recommended
configuration freezes the vision encoder and vocab embeddings (dead weight for value
prediction) while training the Gemma text layers and value head (~315M trainable
params out of ~990M total).

```bash
uv run python -m lerobot.rl.algorithms.RECAPTrainValueNetwork \
  --repo_id="jackvial/so101_pickplace_recap_merged_v2" \
  --root="${HOME}/.cache/huggingface/lerobot" \
  --output_dir="${HOME}/code/lerobot/outputs/so101_pickplace_recap_merged_v2_value_5" \
  --pretrained_path="lerobot/pi05_base" \
  --epochs=2 \
  --batch_size=1 \
  --learning_rate=3e-3 \
  --num_workers=4 \
  --val_split_ratio=0.1 \
  --log_every_n_steps=10 \
  --validate_every_n_train_steps=50 \
  --plot_every_n_train_steps=200 \
  --max_val_steps_per_step_validation=20 \
  --c_fail=500.0 \
  --num_value_bins=56 \
  --val_plot_num_episodes=4 \
  --val_plot_num_frames=8 \
  --val_plot_every_n_epochs=1 \
  --model_precision="bfloat16" \
  --freeze_vision_encoder=true \
  --freeze_embeddings=true \
  --freeze_backbone=false
```

### 2.2 Quick start — SmolVLA backbone (RTX 4070 TI SUPER)

Uses SmolVLM2-500M as the vision-language backbone. The recommended configuration is
to load pretrained SmolVLM2 weights (`--load_vlm_weights=true`), freeze the vision
encoder (`--freeze_vision_encoder=true`), and train the language backbone
(`--freeze_backbone=false`). This lets the model leverage pretrained visual features
while adapting the language backbone to the value prediction task. SmolVLA is smaller
than pi0.5, so you can use larger batch sizes and a higher learning rate.

Key SmolVLA-specific flags:

| Flag | Effect |
|------|--------|
| `--load_vlm_weights` | Load pretrained SmolVLM2 weights (`true`) or random init (`false`) |
| `--freeze_vision_encoder` | Freeze the SigLIP vision encoder inside SmolVLM2 |
| `--freeze_backbone` | Freeze the full SmolVLM2 language backbone (only the value head trains) |

```bash
uv run python -m lerobot.rl.algorithms.RECAPTrainSmolVLANetwork \
  --repo_id="jackvial/so101_pickplace_recap_merged_v2" \
  --root="${HOME}/.cache/huggingface/lerobot" \
  --output_dir="${HOME}/code/lerobot/outputs/so101_pickplace_recap_smolvla_scratch_3" \
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
  --load_vlm_weights=true \
  --freeze_vision_encoder=true \
  --freeze_backbone=false \
  --model_precision="bfloat16"
```

### 2.3 Full smoke-test command (step-based val + plots, pi0.5)

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