# RECAP Value Network Training

This document explains how to:

1. Prepare a LeRobot dataset + success/fail labels for value training.
2. Train the standalone RECAP value network.
3. Train the advantage-conditioned policy (SmolStar06) using the frozen value network.

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
  --output_dir="${HOME}/code/lerobot/outputs/so101_pickplace_recap_value" \
  --epochs=2 \
  --batch_size=6 \
  --learning_rate=3e-3 \
  --num_workers=4 \
  --val_split_ratio=0.1 \
  --log_every_n_steps=100 \
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

---

## 3) Advantage-Conditioned Policy Training (SmolStar06)

The RECAP pipeline has two phases:

1. **Train a value network** (Sections 2.1–2.2 above) — produces a critic that
   predicts expected returns from observations.
2. **Train an advantage-conditioned policy** (this section) — uses the frozen
   value network to label each training sample with an advantage indicator
   ("Advantage: positive" or "Advantage: negative") that is appended to the
   language prompt before standard SmolVLA flow-matching training.

At inference, the model simply conditions on "Advantage: positive" to produce
higher-quality actions. No value network is needed at test time.

### 3.1 How advantage conditioning works

For each training sample `(o_t, a_t)`:

1. The deterministic return `R_t` is computed from the episode success/fail
   label and the frame's position within the episode.
2. The frozen value network predicts `V(o_t)`.
3. Advantage `A = R_t - V(o_t)` measures whether the trajectory did better or
   worse than expected.
4. `A > 0` → append `"Advantage: positive"` tokens to the prompt.
   `A ≤ 0` → append `"Advantage: negative"` tokens.
5. 30% of the time the advantage indicator is dropped entirely (enables
   optional classifier-free guidance at test time with `cfg_beta > 1`).

### 3.2 Quick start — SmolStar06 (RTX 4070 TI SUPER)

Prerequisites:
- A trained SmolVLA value network checkpoint (from Section 2.2). The example
  below uses `outputs/so101_pickplace_recap_smolvla_scratch_3`.
- The same dataset and episode labels used for value network training.

```bash
lerobot-train \
  --policy.type=smolstar06 \
  --policy.value_network_checkpoint="${HOME}/code/lerobot/outputs/so101_pickplace_recap_value/checkpoints/last.pt" \
  --policy.episode_labels_path="${HOME}/.cache/huggingface/lerobot/jackvial/so101_pickplace_recap_merged_v2/meta/episode_labels.csv" \
  --policy.c_fail=500.0 \
  --policy.advantage_threshold=0.0 \
  --policy.advantage_dropout=0.3 \
  --policy.cfg_beta=1.0 \
  --policy.tokenizer_max_length=64 \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=30000 \
  --dataset.repo_id=jackvial/so101_pickplace_recap_merged_v2 \
  --batch_size=6 \
  --policy.push_to_hub=false \
  --save_freq=1000 \
  --steps=10000
```

### 3.3 SmolStar06-specific configuration flags

| Flag | Default | Description |
|------|---------|-------------|
| `value_network_checkpoint` | `None` | Path to `.pt` checkpoint from RECAP value network training |
| `episode_labels_path` | `None` | Path to `episode_labels.csv` with per-episode success/fail labels |
| `c_fail` | `500.0` | Failure penalty (must match the value used for value network training) |
| `advantage_threshold` | `0.0` | Binarization threshold: advantage > threshold → positive |
| `advantage_dropout` | `0.3` | Probability of omitting the advantage indicator (for CFG training) |
| `cfg_beta` | `1.0` | Classifier-free guidance scale at inference (1.0 = no CFG) |
| `tokenizer_max_length` | `64` | Increased from SmolVLA's 48 to accommodate advantage tokens |

### 3.4 Inference

At inference time, no value network is needed:

- **`cfg_beta=1.0`** (default): The model appends "Advantage: positive" to
  every prompt and runs standard SmolVLA action sampling.
- **`cfg_beta>1.0`** (optional): Classifier-free guidance runs two denoising
  passes per Euler step — one conditioned on "positive" and one without the
  indicator — then interpolates the flow vectors for sharper action
  distributions.