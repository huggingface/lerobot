# FastWAM

FastWAM is a World Action Model policy for robot control. The LeRobot integration exposes FastWAM through the standard policy API so it can be configured with `policy.type=fastwam`, trained with `lerobot-train`, and loaded through the LeRobot pretrained policy interface.

## Model Overview

FastWAM keeps video modeling during training, but uses direct action prediction at inference time instead of iteratively generating future observations. This LeRobot policy wraps the FastWAM action model, adapts LeRobot batches to FastWAM training samples, and provides the standard processor pipeline for normalization and action postprocessing.

The implementation initializes the visual world-model components from `Wan-AI/Wan2.2-TI2V-5B` by default and predicts action chunks with shape `[batch, action_horizon, action_dim]`.

### What the LeRobot Integration Covers

- Standard `policy.type=fastwam` configuration through LeRobot
- Image, state, action, and language-task batch adaptation
- Action chunk inference through `select_action` and `predict_action_chunk`
- Checkpoint save/load through the LeRobot policy APIs
- Configurable LIBERO gripper action postprocessing

## Installation Requirements

Install LeRobot from source, then install FastWAM dependencies:

```bash
pip install -e ".[fastwam]"
```

This installs the FastWAM policy extra from `pyproject.toml`: `transformers`,
`diffusers`, `ftfy`, and `regex`, plus LeRobot's base dependencies.

For LIBERO evaluation, install the benchmark dependencies too:

```bash
pip install -e ".[fastwam,libero]"
```

This installs both extras. In addition to the FastWAM dependencies above, the
`libero` extra installs LeRobot dataset dependencies, `hf-libero` on Linux, and
`scipy`.

FastWAM uses the Wan2.2 TI2V backbone. The default model id is:

```python
policy.model_id=Wan-AI/Wan2.2-TI2V-5B
```

## Data Requirements

FastWAM expects a LeRobot dataset with:

- one or more visual observations whose widths concatenate to `policy.image_size[1]`
- `observation.state` when `policy.proprio_dim` is not `None`
- `action`
- a language task instruction through the dataset task field, or precomputed `context` and `context_mask` tensors

The default visual setup is one image feature named `observation.images.image` with shape `(3, 224, 448)`. If the dataset uses two cameras, configure `policy.input_features` so their heights match `224` and their widths sum to `448`.

## Usage

Create a new FastWAM policy with:

```bash
lerobot-train \
  --dataset.repo_id=your-org/your-dataset \
  --policy.type=fastwam \
  --policy.action_dim=7 \
  --policy.proprio_dim=8 \
  --policy.action_horizon=32 \
  --policy.n_action_steps=10 \
  --policy.image_size='[224,448]' \
  --output_dir=./outputs/fastwam_training \
  --job_name=fastwam_training \
  --steps=300000 \
  --batch_size=8 \
  --policy.device=cuda
```

Evaluate an existing LeRobot-format checkpoint on LIBERO-10 with:

```bash
lerobot-eval \
  --policy.path=ZibinDong/fastwam_libero_uncond_2cam224 \
  --policy.device=cuda \
  --policy.torch_dtype=float32 \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_10 \
  --env.observation_height=224 \
  --env.observation_width=224 \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=0 \
  --env.episode_length=600
```

For `libero_goal`, `libero_spatial`, and `libero_object`, use
`--env.episode_length=300`.

For real-robot rollout, use the same checkpoint path:

```bash
lerobot-rollout \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --policy.path=your-org/fastwam-real-robot
```

## Configuration Notes

### Image Features

`policy.image_size` is the size of the concatenated FastWAM image tensor as `(height, width)`. Each configured image feature must have shape `(3, height, camera_width)`, and all camera widths must sum to the configured width.

### Action Chunking

`policy.action_horizon` controls the number of future actions supervised during training and predicted during inference. `policy.n_action_steps` controls how many actions are consumed before the policy predicts a fresh chunk. `policy.n_action_steps` must be less than or equal to `policy.action_horizon`.

### Wan Components

FastWAM loads the Wan VAE, video DiT, text encoder, and tokenizer from the configured Wan model directory or Hugging Face Hub model id. LeRobot-format FastWAM checkpoints saved by `save_pretrained` also copy the local Wan component files needed by `from_pretrained`.

### Attention Backend

FastWAM's DiT uses PyTorch's `scaled_dot_product_attention` (SDPA) for all attention. It does **not** use FlashAttention: its Mixture-of-Transformers (MoT) routing needs arbitrary boolean `[query, key]` attention masks, which the FlashAttention varlen API cannot express. Installing the `flash-attn` package therefore has no effect on the FastWAM path. (Note that SDPA itself may still select PyTorch's own flash / memory-efficient / math kernel internally — this is unrelated to the `flash-attn` package.)

### LIBERO Action Toggle

FastWAM LIBERO checkpoints use `policy.toggle_action_dimensions=[-1]` by
default to match the gripper action convention used by the original FastWAM
evaluation pipeline:

```bash
--policy.toggle_action_dimensions='[-1]'
```

## Results

Evaluated on LIBERO with [`ZibinDong/fastwam_libero_uncond_2cam224`](https://huggingface.co/ZibinDong/fastwam_libero_uncond_2cam224):

| Suite          | Success rate | n_episodes |
| -------------- | -----------: | ---------: |
| libero_spatial |        97.6% |        500 |
| libero_object  |        99.0% |        500 |
| libero_goal    |        95.0% |        500 |
| libero_10      |        94.0% |        500 |
| **average**    |    **96.4%** |       2000 |

Reproduce: `lerobot-eval --policy.path=ZibinDong/fastwam_libero_uncond_2cam224 --policy.device=cuda --policy.torch_dtype=float32 --policy.n_action_steps=10 --env.type=libero --env.task=libero_spatial --env.observation_height=256 --env.observation_width=256 --eval.batch_size=1 --eval.n_episodes=50 --seed=0 --env.episode_length=300` (1x H20 140 GB).

## References

- [Fast-WAM paper](https://arxiv.org/abs/2603.16666)
- [Fast-WAM project page](https://yuantianyuan01.github.io/FastWAM/)
- [Fast-WAM code](https://github.com/yuantianyuan01/FastWAM)
- [Released upstream checkpoints](https://huggingface.co/yuanty/fastwam)
- [Wan2.2 TI2V 5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)

## Citation

```bibtex
@article{yuan2026fastwam,
  title = {Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author = {Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal = {arXiv preprint arXiv:2603.16666},
  year = {2026},
  url = {https://arxiv.org/abs/2603.16666}
}
```
