# LingBot-VA

LingBot-VA is an **autoregressive video-action world-model policy** built on the **Wan2.2**
video-diffusion stack. It interleaves, in one autoregressive sequence, the prediction of
future **video latents** and **robot actions** ("VA" = Video-Action). The LeRobot
integration wires LingBot-VA into the standard training, evaluation and processor
interfaces.

## Model Overview

LingBot-VA is a **dual-stream "mixture-of-transformers"**: a video/latent stream
(`patch_embedding_mlp → blocks → proj_out`) and an action stream
(`action_embedder → blocks → action_proj_out`) share the same 30 transformer blocks and
text conditioning.

| Component                | Class                   | Role                                                        |
| ------------------------ | ----------------------- | ----------------------------------------------------------- |
| DiT backbone (trainable) | `WanTransformer3DModel` | ~5B-param dual-stream transformer.                          |
| VAE (frozen)             | `AutoencoderKLWan`      | Wan2.2 VAE, `z_dim=48`. Lazy-pulled from the source repo.   |
| Text encoder (frozen)    | `UMT5EncoderModel`      | UMT5-XXL, `d_model=4096`. Lazy-pulled from the source repo. |

At inference the policy runs an autoregressive loop per chunk: it denoises the video-latent
stream (CFG, ~20 steps) and the action stream (~50 steps) with two independent
flow-matching schedulers, maintaining a KV cache across chunks. Real observed keyframes are
fed back into the KV cache as the chunk is executed (closed-loop world modeling).

### What the LeRobot Integration Covers

- Standard `policy.type=lingbot_va` configuration through LeRobot.
- Ready-to-use LeRobot-format checkpoints on the Hub (converted from the released upstream ones).
- Autoregressive dual-stream inference behind the standard `select_action` interface
  (single-environment eval, `--eval.batch_size=1`).
- Opt-in saving of the policy's **predicted (imagined) videos** during eval / training.
- Evaluation with `lerobot-eval` on LIBERO and RoboTwin.
- Training / fine-tuning via the dual-stream flow-matching loss (`policy.forward`), see below.

## Installation

1. Install LeRobot by following the [Installation Guide](./installation).
2. Install the LingBot-VA extra:

```bash
pip install -e ".[lingbot_va]"
```

## Checkpoints

The released upstream checkpoints have been converted to LeRobot format and pushed to the Hub:

| Variant                | LeRobot checkpoint               |
| ---------------------- | -------------------------------- |
| LIBERO-Long post-train | `lerobot/lingbot_va_libero_long` |
| RoboTwin post-train    | `lerobot/lingbot_va_robotwin`    |
| Pretrained base        | `lerobot/lingbot_va_base`        |

Only the trainable ~5B transformer is stored in the LeRobot
`model.safetensors`. The frozen VAE + UMT5 + tokenizer (~20 GB) are pulled from
`config.wan_pretrained_path` at load time (defaults to the source `robbyant/*` repo). The
UMT5-XXL text encoder runs on CPU by default (`config.text_encoder_device`) so the 5B
transformer + VAE fit on a single 24–32 GB GPU.

## Evaluation (LIBERO)

```bash
lerobot-eval \
    --policy.path=lerobot/lingbot_va_libero_long \
    --policy.device=cuda \
    --env.type=libero --env.task=libero_10 \
    --env.observation_height=128 --env.observation_width=128 \
    --eval.n_episodes=50 --eval.batch_size=1 \
    --output_dir=outputs/eval/lingbot_va_libero
```

LingBot-VA's streaming inference (KV cache + observed-keyframe feedback) is implemented for
single-environment eval; use `--eval.batch_size=1`.

## Evaluation (RoboTwin)

RoboTwin 2.0 needs the SAPIEN + CuRobo simulator stack. You can use the benchmark Docker image
(`docker/Dockerfile.benchmark.robotwin`, which also needs `warp-lang==1.3.1` and CuRobo built
with the GPU's compute capability in `TORCH_CUDA_ARCH_LIST`). RoboTwin uses **end-effector-pose
control**, so run with `--env.action_mode=ee`: the policy predicts per-arm `xyz+quaternion+gripper`
deltas (`robotwin_tshape` latent layout) that are composed onto the episode's initial eef pose and
executed via CuRobo IK.

```bash
lerobot-eval \
    --policy.path=lerobot/lingbot_va_robotwin \
    --policy.device=cuda \
    --env.type=robotwin --env.task=beat_block_hammer --env.action_mode=ee \
    --eval.n_episodes=10 --eval.batch_size=1 \
    --output_dir=outputs/eval/lingbot_va_robotwin
```

### Saving predicted (imagined) videos

Set `--policy.save_predicted_video=true` to additionally VAE-decode the predicted video
latents and write `pred_episode_*.mp4` next to the env-rendered `eval_episode_*.mp4` videos.
The same flag works for the periodic eval during `lerobot-train`.

## Training / fine-tuning

`LingBotVAPolicy.forward(batch)` implements the dual-stream **flow-matching** loss
(`latent_loss + action_loss`, timestep-weighted, action-masked) from the paper: it VAE-encodes
the camera clips into video latents, UMT5-encodes the task, noises both streams, runs the
transformer's block-causal training pass and returns `(loss, metrics)`. Optimizer preset is AdamW
with a linear-warmup-then-constant schedule (matching upstream).

Requirements:

- The block-causal masks use PyTorch **flex-attention**, so build the policy with
  `--policy.attn_mode=flex` for training (the default `torch` SDPA is inference-only).
- The full 5B DiT does not fit a single 24–32 GB GPU under AdamW; fine-tune with **LoRA**
  (`--policy.use_peft=true`) and/or optimizer offload. `get_optim_params` returns only the
  trainable (e.g. adapter) parameters; the VAE + UMT5 text encoder stay frozen.

```bash
lerobot-train \
  --policy.path=lerobot/lingbot_va_libero_long --policy.attn_mode=flex \
  --policy.use_peft=true \
  --dataset.repo_id=<your LeRobot-format dataset> \
  --batch_size=1 --steps=... --output_dir=outputs/train/lingbot_va
```

The dataset must provide camera clips (a temporal window per camera, VAE-encoded to
`frame_chunk_size` latent frames) and `frame_chunk_size * action_per_frame` action steps per item.

## Data format (action channels & camera order)

LingBot-VA is an **end-effector (Cartesian) pose** policy, it predicts EEF poses + gripper, not
joint positions. Actions live in a fixed multi-embodiment **30-dim** layout; map your robot's
action dimensions into these channels and pad the rest with `0` (`used_action_channel_ids` selects
the channels a given checkpoint actually uses):

| channels | meaning                                               |
| -------- | ----------------------------------------------------- |
| 0–6      | Left-arm end-effector pose                            |
| 7–13     | Right-arm end-effector pose                           |
| 14–20    | Left-arm joints (unused by the released checkpoints)  |
| 21–27    | Right-arm joints (unused by the released checkpoints) |
| 28       | Left gripper                                          |
| 29       | Right gripper                                         |

- **LIBERO** uses channels `0–6`: a 6-DoF EEF delta (xyz + rotation) + gripper (single arm).
- **RoboTwin** uses channels `[0–6, 28, 7–13, 29]`: left EEF (xyz + quaternion) + left gripper +
  right EEF + right gripper (16 dims). The env converts these poses to joint trajectories via
  CuRobo IK — joints are never predicted.

Joint-space datasets (or a different EEF convention) must be remapped into this schema before
fine-tuning these checkpoints.

**Camera order is fixed and order-sensitive**, per-camera latents are concatenated spatially in
`obs_cam_keys` order, so the physical camera→slot mapping must match training:

| benchmark | `obs_cam_keys` (in order)                                                                             | `camera_layout`                                                     |
| --------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| LIBERO    | `observation.images.image` (agentview / 3rd-person), `observation.images.image2` (eye-in-hand wrist)  | `width_concat` (latents concatenated on width)                      |
| RoboTwin  | `observation.images.head_camera`, `observation.images.left_camera`, `observation.images.right_camera` | `robotwin_tshape` (full-res head below, two half-res wrists on top) |

The first camera is the exterior/head view and the rest are wrist views.

## Inference Hyperparameters (LIBERO)

| Key                                    | Value                                                                             |
| -------------------------------------- | --------------------------------------------------------------------------------- |
| height × width                         | 128 × 128                                                                         |
| cameras                                | `observation.images.image` (agentview), `observation.images.image2` (eye-in-hand) |
| action channels used                   | 0–6 (7-DoF arm + gripper)                                                         |
| action_per_frame / frame_chunk_size    | 4 / 4                                                                             |
| attn_window                            | 30                                                                                |
| video / action denoising steps         | 20 / 50                                                                           |
| guidance_scale / action_guidance_scale | 5 / 1                                                                             |
| snr_shift / action_snr_shift           | 5.0 / 0.05                                                                        |

These are the defaults of `LingBotVAConfig`; override any of them via `--policy.<name>=...`.

## Notes

- **Attention backend:** inference uses the `torch` SDPA backend (always available). The
  `flashattn` and `flex` backends are optional; `flex` is only needed for training.
- **Model size:** the DiT is ~5B params and the frozen VAE+UMT5 add ~20 GB; inference needs
  roughly 18–24 GB of VRAM.

## License

LingBot-VA is released under Apache-2.0. See the
[upstream repository](https://github.com/Robbyant/lingbot-va).
