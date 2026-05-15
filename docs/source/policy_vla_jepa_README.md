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

The action head is a **Diffusion Transformer (DiT-B)** with flow matching:

- **Inner dim**: 768 (12 heads × 64 head dim, DiT-B preset)
- **Output dim**: `action_hidden_size` (default 1024), projected down to `action_dim`
- **Cross/self alternation**: even-indexed DiT blocks attend to Qwen context tokens (cross-attention); odd-indexed blocks are self-attention
- **Noise schedule**: Beta distribution with parameters `action_noise_beta_alpha` / `action_noise_beta_beta`
- **Inference**: Euler integration over `num_inference_timesteps` steps

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

Three checkpoints are available, converted from [ginwind/VLA-JEPA](https://huggingface.co/ginwind/VLA-JEPA):

| Checkpoint                    | Dataset           | Cameras                 | World model | Action dim |
| ----------------------------- | ----------------- | ----------------------- | ----------- | ---------- |
| `lerobot/VLA-JEPA-LIBERO`     | LIBERO-10         | 2 (agentview + wrist)   | Enabled     | 7          |
| `lerobot/VLA-JEPA-Pretrain`   | DROID 1.0.1       | 2 (exterior left views) | Enabled     | 7          |
| `lerobot/VLA-JEPA-SimplerEnv` | OXE Bridge / RT-1 | 1                       | Disabled\*  | 7          |

\* The SimplerEnv checkpoint was fine-tuned from Pretrain. The world model predictor architecture expects `embed_dim=2048` (2-camera input) but SimplerEnv is single-camera, so the world model cannot be loaded cleanly. Since inference only needs Qwen + the action head, `enable_world_model=False` is set for this variant. See [Fine-tuning on single-camera datasets](#fine-tuning-on-single-camera-datasets) for implications.

All checkpoints use `Qwen/Qwen3-VL-2B-Instruct` as the language backbone.

### Loading a pretrained checkpoint

```python
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

policy = VLAJEPAPolicy.from_pretrained("lerobot/VLA-JEPA-LIBERO")
```

---

## Configuration

Key parameters in `VLAJEPAConfig`:

| Parameter                                    | Default                            | Description                                                         |
| -------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------- |
| `qwen_model_name`                            | `"Qwen/Qwen3-VL-2B-Instruct"`      | Qwen3-VL backbone variant                                           |
| `jepa_encoder_name`                          | `"facebook/vjepa2-vitl-fpc64-256"` | V-JEPA2 video encoder                                               |
| `chunk_size`                                 | 16                                 | Number of actions predicted per inference call                      |
| `n_action_steps`                             | 16                                 | Steps executed from the predicted chunk before re-planning          |
| `num_video_frames`                           | 16                                 | Video clip length fed to the world model                            |
| `jepa_tubelet_size`                          | 2                                  | Temporal patch size of the video encoder (must match encoder)       |
| `action_model_type`                          | `"DiT-B"`                          | DiT preset — controls hidden dim, heads, head dim                   |
| `action_hidden_size`                         | 1024                               | DiT output projection size (and action decoder input size)          |
| `action_num_layers`                          | 12                                 | Number of DiT transformer blocks                                    |
| `num_target_vision_tokens`                   | 32                                 | Learned future-vision query tokens prepended to the action sequence |
| `action_max_seq_len`                         | 1024                               | Max length of the positional embedding table in the action head     |
| `num_action_tokens_per_timestep`             | 4                                  | Special action tokens per temporal step (used for WM conditioning)  |
| `num_embodied_action_tokens_per_instruction` | 8                                  | Instruction-level embodied tokens appended to the Qwen sequence     |
| `num_inference_timesteps`                    | 10                                 | Euler integration steps for action denoising                        |
| `enable_world_model`                         | `True`                             | Whether to load and train the V-JEPA2 predictor                     |
| `world_model_loss_weight`                    | 0.1                                | Weight of the JEPA prediction loss relative to the action loss      |
| `predictor_depth`                            | 6                                  | Number of transformer blocks in the video predictor                 |
| `repeated_diffusion_steps`                   | 4                                  | Independent noise draws per batch item (CogACT-style augmentation)  |

---

## Training

### Full training from scratch

```bash
lerobot-train \
  dataset.repo_id=your_org/your_dataset \
  policy.chunk_size=16 \
  policy.n_action_steps=16
```

### Fine-tuning from a pretrained checkpoint

```bash
lerobot-train \
  policy.path=lerobot/VLA-JEPA-LIBERO \
  dataset.repo_id=your_org/your_dataset
```

### Reproducing the LIBERO results

**Training on LIBERO:**

TODO(Maxime):

- [ ] double check the training command
- [ ] double check which LIBERO dataset (libero_10 or full libero) was used for training the checkpoint
- [ ] add the evaluation command for the pretrained checkpoint + check that the results match the original paper

```bash
lerobot-train \
  policy.path=lerobot/VLA-JEPA-Pretrain \
  dataset.repo_id=lerobot/libero_10 \
  policy.chunk_size=7 \
  policy.n_action_steps=7 \
  policy.future_action_window_size=6 \
  policy.num_video_frames=8 \
  policy.num_action_tokens_per_timestep=8 \
  policy.num_embodied_action_tokens_per_instruction=32 \
  policy.action_num_layers=16 \
  policy.predictor_depth=12 \
  training.num_steps=50000 \
  env.type=libero \
  env.task=libero_10
```

**Evaluating the pretrained LIBERO-10 checkpoint:**

```bash
lerobot-eval \
  --policy.path=lerobot/VLA-JEPA-LIBERO \
  --env.type=libero \
  --env.task=libero_10 \
  --env.obs_type=pixels_agent_pos \
  --eval.n_episodes=500 \
  --eval.batch_size=10 \
  --policy.device=cuda
```

This runs all 10 LIBERO-10 tasks (50 episodes each, 500 total) with the default camera setup (`agentview_image` → `observation.images.image`, `robot0_eye_in_hand_image` → `observation.images.image2`) and the `pixels_agent_pos` obs type that provides both images and robot state.

To evaluate a subset of tasks only:

```bash
lerobot-eval \
  --policy.path=lerobot/VLA-JEPA-LIBERO \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0,1,2]' \
  --eval.n_episodes=50 \
  --eval.batch_size=5 \
  --policy.device=cuda
```

---

## Fine-tuning on single-camera datasets

The pretrained world model predictor was trained with `embed_dim = num_views × 1024`. If your target dataset has fewer cameras than the source checkpoint, the predictor input projection will have a shape mismatch and cannot be loaded.

**Option 1 — Disable the world model (recommended)**

Set `enable_world_model=False`. Only the Qwen backbone and action head are loaded and trained. This matches the original SimplerEnv fine-tuning strategy and is sufficient for good action performance.

```bash
lerobot-train \
  policy.path=lerobot/VLA-JEPA-Pretrain \
  policy.enable_world_model=false \
  dataset.repo_id=your_org/single_camera_dataset
```

**Option 2 — Reinitialize the predictor input projection**

If you want the JEPA self-supervised signal during fine-tuning, load the checkpoint with `strict=False` and reinitialize `model.video_predictor.predictor_embed` for the new `embed_dim`. All other predictor block weights (attention, MLP, norm, output projection) are camera-count-agnostic and can be reused from the pretrained checkpoint.

---

## Citation

```bibtex
@misc{vla_jepa_2025,
  title   = {VLA-JEPA: Vision-Language-Action Model with Joint-Embedding Predictive Architecture},
  author  = {Gin, Wind and others},
  year    = {2025},
  url     = {https://huggingface.co/ginwind/VLA-JEPA},
}
```

---

## License

Weights are distributed under the license terms of the original [ginwind/VLA-JEPA](https://huggingface.co/ginwind/VLA-JEPA) repository. The LeRobot integration code follows the **Apache 2.0 License**.
