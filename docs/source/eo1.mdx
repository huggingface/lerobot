# EO-1

EO-1 is a **Vision-Language-Action policy for robot control**. The LeRobot implementation integrates EO-1 with the standard LeRobot training, evaluation, processor interface.

## Model Overview

EO-1 uses a Qwen2.5-VL backbone for vision-language understanding and adds a continuous flow-matching action head for robot control. The policy formats each robot-control sample as a multimodal conversation: camera images are passed to Qwen2.5-VL, the robot state is represented with EO-1 state tokens, and the future action chunk is represented with EO-1 action tokens.

<img
  src="https://huggingface.co/datasets/HaomingSong/lerobot-documentation-images/resolve/main/lerobot/eo_pipeline.png"
  alt="An overview of EO-1"
  width="85%"
/>

During training, EO-1 learns to denoise continuous action chunks at the action-token positions. During inference, it samples an action chunk, returns continuous actions, and executes `n_action_steps` from the chunk before sampling again.

### What the LeRobot Integration Covers

- Standard `policy.type=eo1` configuration through LeRobot
- Qwen2.5-VL image and text preprocessing through policy processors
- Continuous flow-matching action prediction
- Checkpoint save/load through LeRobot policy APIs
- Training with `lerobot-train` and evaluation with `lerobot-eval`

The broader EO-1 project also includes interleaved vision-text-action pretraining and multimodal reasoning workflows. This page focuses on the LeRobot robot-control policy path.

## Installation Requirements

1. Install LeRobot by following the [Installation Guide](./installation).
2. Install EO-1 dependencies by running:

   ```bash
   pip install -e ".[eo1]"
   ```

3. If you want to train or evaluate on LIBERO, install the LIBERO dependencies too:

   ```bash
   pip install -e ".[eo1,libero]"
   ```

EO-1 can use the standard PyTorch scaled-dot-product attention backend through `policy.attn_implementation=sdpa`. If your environment has a compatible `flash_attn` installation, you can request `policy.attn_implementation=flash_attention_2`.

## Data Requirements

EO-1 expects a LeRobot dataset with:

- At least one visual observation, for example `observation.images.image`
- `observation.state`
- `action`
- A language task instruction through the dataset `task` field

If your dataset uses different observation names, use `rename_map` to align them with the names expected by your training or evaluation setup.

## Usage

To use EO-1 in a LeRobot configuration, specify the policy type as:

```python
policy.type=eo1
```

By default, a new EO-1 policy initializes its backbone from:

```python
policy.vlm_base=Qwen/Qwen2.5-VL-3B-Instruct
```

Once a LeRobot-format EO-1 checkpoint is available, load it with:

```python
policy.path=your-org/your-eo1-checkpoint
```

## Training

### Training Command Example

```bash
lerobot-train \
  --dataset.repo_id=your_org/your_dataset \
  --policy.type=eo1 \
  --policy.vlm_base=Qwen/Qwen2.5-VL-3B-Instruct \
  --policy.dtype=bfloat16 \
  --policy.attn_implementation=sdpa \
  --policy.gradient_checkpointing=false \
  --output_dir=./outputs/eo1_training \
  --job_name=eo1_training \
  --steps=300000 \
  --batch_size=16 \
  --policy.device=cuda
```

### Key Training Parameters

| Parameter                              | Default                       | Description                                                             |
| -------------------------------------- | ----------------------------- | ----------------------------------------------------------------------- |
| `policy.vlm_base`                      | `Qwen/Qwen2.5-VL-3B-Instruct` | Qwen2.5-VL checkpoint used to initialize a new policy                   |
| `policy.dtype`                         | `auto`                        | Backbone dtype request: `auto`, `bfloat16`, or `float32`                |
| `policy.attn_implementation`           | `None`                        | Optional Qwen attention backend, such as `sdpa`                         |
| `policy.gradient_checkpointing`        | `false`                       | Reduces memory usage during training                                    |
| `policy.chunk_size`                    | `8`                           | Number of future actions predicted per chunk                            |
| `policy.n_action_steps`                | `8`                           | Number of actions consumed from a sampled chunk                         |
| `policy.num_denoise_steps`             | `10`                          | Number of flow-matching denoising steps used during sampling            |
| `policy.max_state_dim`                 | `32`                          | State padding dimension                                                 |
| `policy.max_action_dim`                | `32`                          | Action padding dimension                                                |
| `policy.force_fp32_autocast`           | `true`                        | Keeps the flow head in fp32 even when the backbone uses mixed precision |
| `policy.supervise_padding_action_dims` | `true`                        | Controls whether padded action dimensions are supervised                |
| `policy.supervise_padding_actions`     | `true`                        | Controls whether padded future action rows are supervised               |

## Evaluation

EO-1 can be evaluated through `lerobot-eval` once you have a LeRobot-format checkpoint:

```bash
lerobot-eval \
  --policy.path=your-org/your-eo1-checkpoint \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=20
```

For datasets or environments whose camera names differ from the checkpoint configuration, pass a `rename_map`:

```bash
lerobot-eval \
  --policy.path=your-org/your-eo1-checkpoint \
  --env.type=libero \
  --env.task=libero_object \
  --rename_map='{"observation.images.image2":"observation.images.wrist_image"}'
```

## Configuration Notes

### Image Processing

EO-1 uses the Qwen2.5-VL processor. The `policy.image_min_pixels` and `policy.image_max_pixels` settings control the image resizing bounds before the visual tokens are passed into the backbone.

### State and Action Dimensions

The policy pads state and action vectors to `policy.max_state_dim` and `policy.max_action_dim` before the EO-1 flow head. Predictions are cropped back to the original action dimension before being returned by the policy.

### Attention Backend

Use `policy.attn_implementation=sdpa` for a portable setup. Use `flash_attention_2` only when `flash_attn` is installed and compatible with your environment.

## References

- [EO-1 project](https://github.com/EO-Robotics/EO1)
- [EO-1 paper](https://arxiv.org/abs/2508.21112)
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

## Citation

```bibtex
@article{eo1,
  title={EO-1: Interleaved Vision-Text-Action Pretraining for General Robot Control},
  author={Delin Qu and Haoming Song and Qizhi Chen and Zhaoqing Chen and Xianqiang Gao and Xinyi Ye and Qi Lv and Modi Shi and Guanghui Ren and Cheng Ruan and Maoqing Yao and Haoran Yang and Jiacheng Bao and Bin Zhao and Dong Wang},
  journal={arXiv preprint},
  year={2025},
  url={https://arxiv.org/abs/2508.21112}
}
```

## License

This LeRobot integration follows the **Apache 2.0 License** used by LeRobot. Check the upstream EO-1 model and dataset pages for the licenses of released EO-1 checkpoints and data.
