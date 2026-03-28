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
   value network to label each training sample with a binarized advantage
   indicator that is injected as a learned embedding directly into the action
   expert's input pathway.

At inference, the model conditions on the positive advantage embedding to
produce higher-quality actions. No value network is needed at test time.

### 3.1 How advantage conditioning works: RECAP paper vs SmolVLA

The RECAP paper (pi-0.6) and SmolVLA use fundamentally different model
architectures, which means the advantage conditioning must be implemented
differently. Understanding this distinction is critical.

#### 3.1.1 RECAP paper (pi-0.6): single-stream architecture

pi-0.6 uses a **single-stream** architecture where the VLM backbone processes a
unified token sequence. The advantage indicator appears as text tokens in that
sequence, positioned after the sub-task prediction but before the action tokens:

```
[images] [language: "pick up the cube"] [sub-task: "grasp object"] [Advantage: positive] [action tokens]
```

Because the action expert shares the same causal sequence as the language model,
the advantage text is the **most recent context** when the expert generates
actions. Standard causal attention ensures the expert directly attends to the
advantage tokens. At inference, you simply append `"Advantage: positive"` to the
prompt and sample actions normally.

#### 3.1.2 SmolVLA: two-stream architecture

SmolVLA uses a **two-stream** architecture with separate processing stacks:

- **Stream 0 (VLM)**: processes images, language tokens, and robot state
- **Stream 1 (action expert)**: processes noisy actions and flow-matching timestep

The two streams are connected only through **cross-attention** (and joint
attention at every other layer). There is no shared token sequence where
advantage text can be placed "right before the actions."

If you append `"Advantage: positive"` as text tokens to the language prompt, the
advantage signal enters the VLM stream (stream 0) and must survive the entire
VLM transformer stack before reaching the action expert indirectly through
cross-attention to VLM key/value states. In practice:

1. The advantage is 4 tokens among 200+ prefix tokens (images produce many
   tokens) — the signal is too diluted in the VLM's internal representations.
2. The VLM has no training objective that encourages it to amplify the advantage
   signal in its key/value states.
3. The gradient from the flow-matching MSE loss must backpropagate through the
   expert's cross-attention, through the VLM's K/V projections, through every
   VLM layer, back to the advantage token embeddings — an extremely weak signal.
4. **Result**: the model produces identical flow-matching loss regardless of
   advantage label (`cond_acc=0.0`, `cond_gap=0.0`).

#### 3.1.3 SmolStar06 solution: expert-side advantage embedding

SmolStar06 injects the advantage signal as a **learned embedding** directly into
the action expert's input pathway (`embed_suffix`), bypassing the VLM text
processing entirely:

```
nn.Embedding(2, expert_hidden_size)  →  index 0 = negative, index 1 = positive
```

The embedding vector is added to the action-time embedding in `embed_suffix`,
so the information path is:

```
advantage_indicator (True/False)
    → nn.Embedding lookup → dense vector [expert_hidden_size]
    → broadcast to [B, chunk_size, expert_hidden_size]
    → ADD to action_time_emb in embed_suffix
    → expert layers process this directly
    → action_out_proj → predicted velocity v_t
    → MSE loss against target u_t
```

The gradient path is **direct**: MSE loss → `action_out_proj` → expert layers →
advantage embedding weights. The embedding is zero-initialized so it starts
neutral, and the MSE loss immediately provides gradient signal.

The VLM prefix (images, language, state) is processed normally with no advantage
tokens — the expert still gets all language/visual information through
cross-attention, exactly as in base SmolVLA. The advantage embedding is a
**separate, dedicated channel** that does not compete with image and language
tokens for attention bandwidth.

### 3.2 Training

For each training sample `(o_t, a_t)`:

1. The deterministic return `R_t` is computed from the episode success/fail
   label and the frame's position within the episode.
2. The frozen value network predicts `V(o_t)`.
3. Advantage `A = R_t - V(o_t)` measures whether the trajectory did better or
   worse than expected.
4. `A > threshold` → advantage embedding index 1 (positive).
   `A ≤ threshold` → advantage embedding index 0 (negative).
5. The advantage embedding vector is added to the action-time embedding in
   `embed_suffix`, directly modulating the expert's denoising input.
6. With 30% probability, the advantage embedding is zeroed out (dropout),
   producing an unconditional forward pass. This enables optional
   classifier-free guidance at test time with `cfg_beta > 1`.

### 3.3 Quick start — SmolStar06 (RTX 4070 TI SUPER)

Prerequisites:
- A trained SmolVLA value network checkpoint (from Section 2.2). The example
  below uses `outputs/so101_pickplace_recap_value`.
- The same dataset and episode labels used for value network training.

```bash
uv run python -m lerobot.rl.algorithms.RECAPTrainSmolStar \
  --repo_id="jackvial/so101_pickplace_recap_merged_v2" \
  --output_dir="${HOME}/code/lerobot/outputs/recap_smolstar_train_4" \
  --value_network_checkpoint="${HOME}/code/lerobot/outputs/so101_pickplace_recap_value/checkpoints/last.pt" \
  --epochs=5 \
  --batch_size=6 \
  --learning_rate=1e-4 \
  --val_split_ratio=0.1 \
  --validate_every_n_train_steps=200 \
  --c_fail=500.0 \
  --advantage_threshold=0.0 \
  --advantage_dropout=0.3 \
  --log_every_n_steps=10
```

### 3.4 SmolStar06-specific configuration flags

| Flag | Default | Description |
|------|---------|-------------|
| `value_network_checkpoint` | `None` | Path to `.pt` checkpoint from RECAP value network training |
| `episode_labels_path` | `None` | Path to `episode_labels.csv` with per-episode success/fail labels |
| `c_fail` | `500.0` | Failure penalty (must match the value used for value network training) |
| `advantage_threshold` | `0.0` | Binarization threshold: advantage > threshold → positive |
| `advantage_dropout` | `0.3` | Probability of zeroing out the advantage embedding (for CFG training) |
| `cfg_beta` | `1.0` | Classifier-free guidance scale at inference (1.0 = no CFG) |

---

## 4) Weights & Biases Logging

Both the value network and SmolStar06 training scripts support optional
[Weights & Biases](https://wandb.ai/) logging. When enabled, all metrics that
are logged to the console and saved to `metrics_history.json` are also sent to
a W&B run in real time.

### 4.1 Enabling W&B

Pass `--wandb_project` to either training script. If omitted (the default),
W&B is completely disabled — no import, no network calls.

| Flag | Default | Description |
|------|---------|-------------|
| `--wandb_project` | `None` | W&B project name. Setting this enables logging. |
| `--wandb_entity` | `None` | W&B team/user entity (uses your default if unset). |
| `--wandb_run_name` | `None` | Display name for the run (auto-generated if unset). |

### 4.2 Metrics logged — value network

| Key prefix | Metrics | When |
|------------|---------|------|
| `train/` | `loss`, `bin_acc`, `value_mae`, `window_loss`, `window_bin_acc`, `window_value_mae`, `samples_per_sec` | Every `log_every_n_steps` training steps |
| `val/` | `loss`, `bin_acc`, `value_mae` | Every validation (step-based or epoch-end) |
| `epoch/` | `train_loss`, `train_bin_acc`, `train_value_mae`, `val_loss`, `val_bin_acc`, `val_value_mae`, `lr` | End of each epoch |
| `val_plots/` | Episode return plots (as `wandb.Image`) | When validation plots are saved |

### 4.3 Metrics logged — SmolStar06 policy

| Key prefix | Metrics | When |
|------------|---------|------|
| `train/` | `loss`, `lr`, `step_loss` | Every `log_every_n_steps` training steps |
| `val/` | `val_loss`, `val_loss_pos`, `val_loss_neg`, `val_n_pos`, `val_n_neg`, `val_conditioning_accuracy`, `val_conditioning_gap`, `val_conditioning_gap_pos`, `val_conditioning_gap_neg`, `val_adv_episode_alignment`, `val_alignment_on_success`, `val_alignment_on_failure` | Every validation (step-based or epoch-end) |
| `epoch/` | All of the above plus `epoch`, `train_loss`, `lr` | End of each epoch |

### 4.4 Example commands

Value network with W&B:

```bash
uv run python -m lerobot.rl.algorithms.RECAPTrainValueNetwork \
  --repo_id="jackvial/so101_pickplace_recap_merged_v2" \
  --output_dir="${HOME}/code/lerobot/outputs/recap_value_wandb" \
  --epochs=2 \
  --batch_size=1 \
  --wandb_project="recap-value-network" \
  --wandb_entity="my-team" \
  --wandb_run_name="value-net-run-1"
```

SmolStar06 policy with W&B:

```bash
uv run python -m lerobot.rl.algorithms.RECAPTrainSmolStar \
  --repo_id="jackvial/so101_pickplace_recap_merged_v2" \
  --output_dir="${HOME}/code/lerobot/outputs/recap_smolstar_train_5" \
  --value_network_checkpoint="${HOME}/code/lerobot/outputs/so101_pickplace_recap_value/checkpoints/last.pt" \
  --epochs=5 \
  --batch_size=6 \
  --learning_rate=1e-3 \
  --val_split_ratio=0.1 \
  --validate_every_n_train_steps=50 \
  --c_fail=500.0 \
  --advantage_threshold=0.0 \
  --advantage_dropout=0.3 \
  --log_every_n_steps=10 \
  --wandb_project="recap-smolstar" \
  --wandb_run_name="smolstar-run-5"
```

---

### 3.5 Inference

At inference time, no value network is needed:

- **`cfg_beta=1.0`** (default): The positive advantage embedding (index 1) is
  added to every `embed_suffix` call during denoising. Language tokens are
  unmodified. This produces actions conditioned on "better than average."
- **`cfg_beta>1.0`** (optional): Classifier-free guidance runs two denoising
  passes per Euler step — one with the positive advantage embedding
  (conditioned) and one without any advantage embedding (unconditional) — then
  interpolates the flow vectors:
  `v = v_uncond + beta * (v_cond - v_uncond)`.