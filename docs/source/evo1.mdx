# EVO1

EVO1 is a Vision-Language-Action policy for robot control built around an InternVL3 backbone and a continuous flow-matching action head. This LeRobot integration exposes EVO1 as a standard policy type so it can be trained and evaluated with the usual LeRobot dataset, checkpoint, and processor APIs.

## Model Overview

The policy embeds one or more camera images and the language task prompt with InternVL3, pads robot state/action vectors to fixed maximum dimensions, and predicts future action chunks with a flow-matching action head. During inference, the policy samples an action chunk and returns `n_action_steps` actions from that chunk before sampling again.

### What the LeRobot Integration Covers

- Standard `policy.type=evo1` configuration through LeRobot
- InternVL3 image/text embedding with optional FlashAttention fallback
- Stage-based finetuning controls for action-head-only and VLM finetuning runs
- Continuous flow-matching action prediction
- Checkpoint save/load through LeRobot policy APIs
- Training with `lerobot-train` and evaluation with standard policy inference APIs

The broader EVO1 project may include additional training scripts and dataset tooling. This page focuses on the LeRobot robot-control policy path.

## Installation Requirements

1. Install LeRobot by following the [Installation Guide](./installation).
2. Install EVO1 dependencies:

   ```bash
   pip install -e ".[evo1]"
   ```

   For LIBERO evaluation, install the LIBERO extra as well:

   ```bash
   pip install -e ".[evo1,libero]"
   ```

3. Install a `flash-attn` wheel only if it is compatible with your Python, PyTorch, CUDA, and GPU stack. EVO1 falls back to standard attention when `flash_attn` is not available.

EVO1 uses the native Hugging Face `transformers` InternVL implementation, so `policy.vlm_model_name` must point to a natively converted checkpoint such as `OpenGVLab/InternVL3-1B-hf` (note the `-hf` suffix). The first run may download the configured VLM checkpoint unless `policy.vlm_model_name` points to a local model directory.

## Data Requirements

EVO1 expects a LeRobot dataset with:

- One to `policy.max_views` visual observations, for example `observation.images.image`
- `observation.state`
- `action`
- A language task instruction in the dataset `task` field, or another field configured with `policy.task_field`

State and action vectors are padded to `policy.max_state_dim` and `policy.max_action_dim`. Predictions are cropped back to the dataset action dimension before being returned.

## Usage

To use EVO1 in a LeRobot configuration, specify:

```python
policy.type=evo1
```

By default, a new EVO1 policy initializes its VLM from:

```python
policy.vlm_model_name=OpenGVLab/InternVL3-1B-hf
```

Once a LeRobot-format EVO1 checkpoint is available, load it with:

```python
policy.path=your-org/your-evo1-checkpoint
```

## Training

### Stage 1

Stage 1 freezes the VLM and trains the action head:

```bash
lerobot-train \
  --dataset.repo_id=your_org/your_dataset \
  --policy.type=evo1 \
  --policy.training_stage=stage1 \
  --policy.vlm_model_name=OpenGVLab/InternVL3-1B-hf \
  --policy.device=cuda \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.max_state_dim=24 \
  --policy.max_action_dim=24 \
  --policy.optimizer_lr=1e-5 \
  --batch_size=4 \
  --steps=5000 \
  --output_dir=./outputs/evo1_stage1
```

### Stage 2

Stage 2 finetunes the VLM branches and action head. A common workflow starts from a Stage 1 checkpoint:

```bash
lerobot-train \
  --dataset.repo_id=your_org/your_dataset \
  --policy.path=./outputs/evo1_stage1/checkpoints/005000/pretrained_model \
  --policy.training_stage=stage2 \
  --policy.vlm_model_name=OpenGVLab/InternVL3-1B-hf \
  --policy.device=cuda \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.max_state_dim=24 \
  --policy.max_action_dim=24 \
  --policy.optimizer_lr=1e-5 \
  --batch_size=4 \
  --steps=80000 \
  --output_dir=./outputs/evo1_stage2
```

By default, `policy.training_stage` reapplies the finetuning defaults for that stage. This is important when
starting Stage 2 from a Stage 1 checkpoint, because the Stage 1 checkpoint config stores the VLM finetuning
flags as disabled. These stage defaults take precedence over saved or manually supplied `policy.finetune_*`
flags unless `policy.apply_training_stage_defaults=false`, so set that flag only when manually controlling
every finetuning flag.

### Key Training Parameters

| Parameter                                     | Default                     | Description                                                       |
| --------------------------------------------- | --------------------------- | ----------------------------------------------------------------- |
| `policy.vlm_model_name`                       | `OpenGVLab/InternVL3-1B-hf` | Natively converted InternVL3 checkpoint or local model directory  |
| `policy.training_stage`                       | `stage1`                    | `stage1` trains the action head; `stage2` finetunes VLM branches  |
| `policy.apply_training_stage_defaults`        | `true`                      | Reapplies stage finetuning defaults after loading a checkpoint    |
| `policy.vlm_num_layers`                       | `14`                        | Number of InternVL3 language layers kept for the policy           |
| `policy.vlm_dtype`                            | `bfloat16`                  | Requested VLM dtype                                               |
| `policy.use_flash_attn`                       | `true`                      | Requests FlashAttention when installed; otherwise falls back      |
| `policy.enable_gradient_checkpointing`        | `true`                      | Enables checkpointing on supported InternVL3 modules              |
| `policy.gradient_checkpointing_use_reentrant` | `false`                     | Reentrant setting passed to gradient checkpointing when supported |
| `policy.chunk_size`                           | `50`                        | Number of future actions predicted per chunk                      |
| `policy.n_action_steps`                       | `50`                        | Number of actions consumed from a sampled chunk                   |
| `policy.max_state_dim`                        | `24`                        | State padding dimension                                           |
| `policy.max_action_dim`                       | `24`                        | Action padding dimension                                          |
| `policy.postprocess_action_dim`               | `null`                      | Optional action dimension returned after EVO1 postprocessing      |
| `policy.binarize_gripper`                     | `false`                     | Binarizes the postprocessed gripper channel for LIBERO-style eval |
| `policy.task_field`                           | `task`                      | Batch field used as the language prompt                           |

## Inference

Try it out with a trained EVO1 checkpoint:

```bash
lerobot-rollout \
  --policy.path=your-org/your-evo1-checkpoint \
  --inference.type=rtc \ # optional
  ...
```

## Results

### LIBERO Evaluation

> [!NOTE]
> Benchmark results for a `lerobot`-hosted LIBERO checkpoint trained with this implementation
> will be added once training completes.

The official EVO1 LIBERO rollout protocol uses the raw LIBERO camera feature names
(`observation.images.agentview_image` and `observation.images.robot0_eye_in_hand_image`), replans every
14 actions, and binarizes the gripper command before stepping the simulator. The EVO1 policy postprocessor
can crop the padded 24D action back to the 7D LIBERO action space and apply that gripper binarization. To
evaluate a LIBERO checkpoint under the same one-episode-per-task setting, keep the raw camera names instead
of the default `image`/`image2` mapping and set the LIBERO action postprocessing flags:

```bash
lerobot-eval \
  --policy.path=your-org/your-evo1-libero-checkpoint \
  --policy.vlm_model_name=OpenGVLab/InternVL3-1B-hf \
  --policy.device=cuda \
  --policy.use_flash_attn=true \
  --policy.n_action_steps=14 \
  --policy.postprocess_action_dim=7 \
  --policy.binarize_gripper=true \
  --env.type=libero \
  --env.task=libero_object \
  --env.camera_name_mapping="{agentview_image: agentview_image, robot0_eye_in_hand_image: robot0_eye_in_hand_image}" \
  --env.observation_height=448 \
  --env.observation_width=448 \
  --eval.batch_size=1 \
  --eval.n_episodes=1
```

## References

- [EVO1 repository](https://github.com/MINT-SJTU/Evo-1)
- [InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)

## License

This LeRobot integration follows the Apache 2.0 License used by LeRobot. Check the upstream EVO1 and InternVL3 model pages for the licenses of released checkpoints and data.
