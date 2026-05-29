# VLA-JEPA

This is the LeRobot port of **VLA-JEPA**, a Vision-Language-Action model that combines a Qwen3-VL language backbone with a self-supervised video world model (V-JEPA2) and a flow-matching DiT action head.

---

## Architecture Overview

VLA-JEPA has three main components:

| Component               | Module                            | Role                                                    |
| ----------------------- | --------------------------------- | ------------------------------------------------------- |
| **Qwen3-VL backbone**   | `Qwen3VLInterface`                | Fuses images + language instruction into context tokens |
| **DiT-B action head**   | `VLAJEPAActionHead`               | Flow-matching diffusion over the action chunk           |
| **V-JEPA2 world model** | `ActionConditionedVideoPredictor` | Self-supervised video prediction loss (training only)   |

### Data flow

**Training:**

1. A video clip of `num_video_frames` frames is encoded by V-JEPA2 into per-frame patch tokens.
2. The Qwen3-VL backbone processes multi-view images + the task instruction and produces a sequence of context tokens that includes special action tokens (for world model conditioning) and embodied tokens.
3. The action head receives those context tokens as cross-attention keys/values and predicts a denoised action chunk via flow matching.
4. The world model predictor uses the action tokens extracted from Qwen to predict future V-JEPA2 frame embeddings; a regression loss on those predictions is added to the action loss.

**Inference:**
Only Qwen + the action head are used. The world model is not needed at inference time.

### Action head details

Available presets via `action_model_type`:

| Preset  | Hidden dim | Heads | Head dim |
| ------- | ---------- | ----- | -------- |
| `DiT-B` | 768        | 12    | 64       |
| `DiT-L` | 1536       | 32    | 48       |

### World model details

The video predictor is a ViT-style transformer (`ActionConditionedVideoPredictor`) that takes:

- **Frame tokens**: V-JEPA2 patch embeddings projected to `predictor_embed_dim`
- **Action tokens**: Qwen action token embeddings projected to `predictor_embed_dim`

It uses block-causal attention so each temporal step can attend to all previous steps. The predictor's input `embed_dim` equals `num_views × video_encoder_hidden_size` (e.g. 2 views × 1024 = 2048 for the pretrained checkpoints).

---

## Pretrained Checkpoints

Three checkpoints are available directly inside the LeRobot org here: [`lerobot/VLA-JEPA`](https://huggingface.co/collections/lerobot/vla-jepa), converted from [ginwind/VLA-JEPA](https://huggingface.co/ginwind/VLA-JEPA):

| Checkpoint                    | Dataset           | Cameras                 | World model | Action dim |
| ----------------------------- | ----------------- | ----------------------- | ----------- | ---------- |
| `lerobot/VLA-JEPA-LIBERO`     | LIBERO-10         | 2 (agentview + wrist)   | Enabled     | 7          |
| `lerobot/VLA-JEPA-Pretrain`   | DROID 1.0.1       | 2 (exterior left views) | Enabled     | 7          |
| `lerobot/VLA-JEPA-SimplerEnv` | OXE Bridge / RT-1 | 1 (view duplicated ×2)  | Enabled     | 7          |

All checkpoints use `Qwen/Qwen3-VL-2B-Instruct` as the language backbone.

---

## Configuration

Key parameters in `VLAJEPAConfig`:

| Parameter                 | Default | Description                                                                                                                                                                     |
| ------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chunk_size`              | 7       | Number of actions predicted per inference call                                                                                                                                  |
| `n_action_steps`          | 7       | Steps executed from the predicted chunk before re-planning                                                                                                                      |
| `num_video_frames`        | 8       | Video clip length fed to the world model                                                                                                                                        |
| `enable_world_model`      | `True`  | Whether to load and train the V-JEPA2 predictor                                                                                                                                 |
| `world_model_loss_weight` | 0.1     | Weight of the JEPA prediction loss relative to the action loss                                                                                                                  |
| `num_inference_timesteps` | 4       | Euler integration steps for action denoising                                                                                                                                    |
| `freeze_qwen`             | `False` | Freeze the Qwen3-VL backbone and only train the action head                                                                                                                     |
| `reinit_modules`          | `None`  | Key prefixes allowed to be randomly re-initialised on load (for cross-embodiment transfer, see [Fine-tuning on a different embodiment](#fine-tuning-on-a-different-embodiment)) |
| `gripper_dim`             | 6       | Index of the gripper dimension in the action vector (e.g. 6 for a 7-DoF arm with gripper as the last joint)                                                                     |
| `gripper_threshold`       | 0.5     | Threshold used by `pre_snap_gripper_action` and `binarize_gripper_action` to binarize the gripper dimension                                                                     |
| `pre_snap_gripper_action` | `True`  | Snap the gripper dim to {0, 1} before unnormalization. Set to `False` for robots without a binary gripper                                                                       |
| `binarize_gripper_action` | `True`  | Binarize the gripper dim to {-1, 1} after unnormalization. Set to `False` for robots without a binary gripper                                                                   |

---

## Training

Number of training steps may vary based on dataset size and compute budget. The original paper pretrained for 50k on ssv2 + droid jointly, then additional 30k steps for LIBERO, but fewer steps may still yield good performance when fine-tuning from the provided pretrained checkpoints.

### Full training from scratch

```bash
lerobot-train \
  policy.type=vla_jepa \
  policy.repo_id=your_org/your_repo \
  dataset.repo_id=your_org/your_dataset
```

### Fine-tuning from a pretrained checkpoint

```bash
lerobot-train \
  --policy.path=lerobot/VLA-JEPA-Pretrain \
  --policy.repo_id=your_org/your_repo \
  --dataset.repo_id=your_org/your_dataset
```

If you want to freeze the Qwen backbone and only train the action head, set `policy.freeze_qwen=True`:

```bash
lerobot-train \
  --policy.path=lerobot/VLA-JEPA-Pretrain \
  --policy.repo_id=your_org/your_repo \
  --policy.freeze_qwen=true \
  --dataset.repo_id=your_org/your_dataset
```

### Fine-tuning on a different embodiment

When the target robot has a different action or state dimensionality than the pretrained checkpoint, the input/output projection layers of the action head will have mismatched shapes and cannot be loaded directly. `reinit_modules` lets you list the key prefixes that are allowed to mismatch — those layers are randomly re-initialised while every other weight is reused from the checkpoint. Any shape mismatch outside the listed prefixes raises an error.

The layers that depend on `action_dim` and `state_dim` are:

| Layer                                     | Key prefix                          |
| ----------------------------------------- | ----------------------------------- |
| Action encoder (action_dim → inner_dim)   | `model.action_model.action_encoder` |
| Action decoder (hidden_size → action_dim) | `model.action_model.action_decoder` |
| State encoder (state_dim → inner_dim)     | `model.action_model.state_encoder`  |

```bash
lerobot-train \
  --policy.path=lerobot/VLA-JEPA-Pretrain \
  --policy.repo_id=your_org/your_repo \
  --policy.freeze_qwen=true \
  --policy.reinit_modules='["model.action_model.action_encoder", "model.action_model.action_decoder", "model.action_model.state_encoder"]' \
  --dataset.repo_id=your_org/your_dataset
```

If your robot has no proprioceptive state, omit `model.action_model.state_encoder` from the list.

### Reproducing the LIBERO results

**Training on LIBERO:**
starts the training from the Pretrain checkpoint, trains for 30k steps on the LIBERO dataset.
Original paper mentions training across 8 GPUs with a batch size of 32, meaning global batch size of 256.

```bash
lerobot-train \
  --policy.path=lerobot/VLA-JEPA-Pretrain \
  --policy.repo_id=your_org/your_repo \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --steps=30000
```

**Evaluating the pretrained LIBERO-10 checkpoint:**

```bash
lerobot-eval \
  --policy.path=lerobot/VLA-JEPA-LIBERO \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.n_episodes=10 \
  --eval.batch_size=5
```

To evaluate a subset of tasks only:

```bash
lerobot-eval \
  --policy.path=lerobot/VLA-JEPA-LIBERO \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0,1,2]' \
  --eval.n_episodes=10 \
  --eval.batch_size=5
```

**Expected results:**

| Suite          | Episodes | Successes | Success Rate |
| -------------- | -------- | --------- | ------------ |
| libero_spatial | 100      | 93        | **95.0%**    |
| libero_object  | 100      | 100       | **100.0%**   |
| libero_goal    | 100      | 98        | **98.0%**    |
| libero_10      | 100      | 96        | **93.0%**    |
| **Overall**    | **400**  | **387**   | **96.5%**    |

---

## Fine-tuning on datasets with a different number of cameras

The pretrained world model predictor was trained with `embed_dim = jepa_tubelet_size × 1024` (default `jepa_tubelet_size=2`).

**Default behaviour — view padding / trimming (no action required)**

When fine-tuning from `VLA-JEPA-Pretrain` the model automatically adjusts the number of views fed to the world model to match `jepa_tubelet_size`:

- **Single-view datasets (e.g. BridgeV2):** the single-view latent is duplicated to produce a two-view world-model input, preserving the JEPA self-supervised signal without any weight mismatch.
- **>2-view datasets (e.g. DROID with 3 views):** all views are passed to the Qwen backbone (for richer context), but only the first `jepa_tubelet_size` views (one wrist + one third-person, following the configured view order) are used for the world model.

**Option 1 — Disable the world model**

Set `enable_world_model=False` to skip the JEPA loss entirely. Only the Qwen backbone and action head are loaded and trained. This is sufficient for good action performance.

```bash
lerobot-train \
  --policy.path=lerobot/VLA-JEPA-Pretrain \
  --policy.enable_world_model=false \
  --policy.repo_id=your_org/your_repo \
  --dataset.repo_id=your_org/single_camera_dataset
```

**Option 2 — Reinitialize the predictor input projection**

If you want to change `jepa_tubelet_size` to a value other than 2, load the checkpoint with `strict=False` and reinitialize `model.video_predictor.predictor_embed` for the new `embed_dim`. All other predictor block weights (attention, MLP, norm, output projection) are camera-count-agnostic and can be reused from the pretrained checkpoint.

---

## Citation

```bibtex
@misc{sun2026vlajepaenhancingvisionlanguageactionmodel,
  title         = {VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model},
  author        = {Jingwen Sun and Wenyao Zhang and Zekun Qi and Shaojie Ren and Zezhi Liu and Hanxin Zhu and Guangzhong Sun and Xin Jin and Zhibo Chen},
  year          = {2026},
  eprint        = {2602.10098},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2602.10098},
}
```

---

## License

Weights are distributed under the license terms of the original [ginwind/VLA-JEPA](https://huggingface.co/ginwind/VLA-JEPA) repository (**Apache 2.0 License**). The LeRobot integration code follows the **Apache 2.0 License**.
