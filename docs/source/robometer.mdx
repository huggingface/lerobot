# ROBOMETER

ROBOMETER is a **general-purpose video-language robotic reward model**. It predicts dense, frame-level task progress and frame-level success from a trajectory video and a task description.

**Paper**: [ROBOMETER: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons](https://arxiv.org/abs/2603.02115)
**Project**: [robometer.github.io](https://robometer.github.io/)
**Original code**: [github.com/robometer/robometer](https://github.com/robometer/robometer)
**Checkpoint**: [lerobot/Robometer-4B](https://huggingface.co/lerobot/Robometer-4B)

## Overview

ROBOMETER builds on `Qwen/Qwen3-VL-4B-Instruct` and adds three lightweight prediction heads:

- **Progress head**: predicts per-frame task progress in `[0, 1]`.
- **Success head**: predicts per-frame task success probability.
- **Preference head**: predicts which of two trajectories better completes the task during training.

The paper trains ROBOMETER with a composite objective:

```text
L = L_pref + L_prog + L_succ
```

The LeRobot integration is currently **inference-only**. It preserves the preference head so that the published `Robometer-4B` checkpoint loads without remapping, but `compute_reward()` queries the progress or success head only.

## What the LeRobot Integration Covers

- Standard `reward_model.type=robometer` configuration through LeRobot.
- Qwen3-VL image and text preprocessing through `RobometerEncoderProcessorStep`.
- LeRobot reward-model save/load APIs through `PreTrainedRewardModel`.
- Dense, frame-level progress and success predictions internally.
- A scalar reward through `compute_reward()` for downstream LeRobot reward-model usage.

This page focuses on using the published ROBOMETER checkpoint as a zero-shot reward model. Training ROBOMETER from scratch is outside the current LeRobot integration.

## Installation Requirements

1. Install LeRobot by following the [Installation Guide](./installation).
2. Install the ROBOMETER dependencies:

```bash
pip install -e ".[robometer]"
```

If you use `uv` directly from a source checkout:

```bash
uv sync --extra robometer
```

ROBOMETER uses a Qwen3-VL-4B backbone, so GPU inference is strongly recommended.

## Model Inputs and Outputs

ROBOMETER expects:

- A trajectory video or sequence of frames.
- A natural-language task description.

In LeRobot datasets, the preprocessor reads:

| Config field              | Default                  | Meaning                                               |
| ------------------------- | ------------------------ | ----------------------------------------------------- |
| `reward_model.image_key`  | `observation.images.top` | Camera/video observation used by ROBOMETER            |
| `reward_model.task_key`   | `task`                   | Key in complementary data that stores the task string |
| `reward_model.max_frames` | `8`                      | Maximum number of frames passed to ROBOMETER          |

The model predicts per-frame progress and success internally. The LeRobot reward API returns a scalar per sample:

- `reward_output="progress"` (default): return the last-frame progress, clamped to `[0, 1]`.
- `reward_output="success"`: return `1.0` if the last-frame success probability is above `success_threshold`, otherwise `0.0`.

## Usage

### Load the Reward Model Directly

```python
from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel

cfg = RobometerConfig(
    pretrained_path="lerobot/Robometer-4B",
    device="cuda",
    reward_output="progress",
)
reward_model = RobometerRewardModel.from_pretrained(cfg.pretrained_path, config=cfg)
```

### Encode Frames and Compute a Reward

For a direct Python call, provide frames as `uint8` arrays with shape `(T, H, W, C)` and a task string:

```python
from lerobot.rewards.robometer.modeling_robometer import ROBOMETER_FEATURE_PREFIX
from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep

# frames: np.ndarray, shape (T, H, W, C), dtype uint8
# task: str
encoder = RobometerEncoderProcessorStep(
    base_model_id=cfg.base_model_id,
    use_multi_image=cfg.use_multi_image,
    use_per_frame_progress_token=cfg.use_per_frame_progress_token,
    max_frames=cfg.max_frames,
)

encoded = encoder.encode_samples([(frames, task)])
batch = {f"{ROBOMETER_FEATURE_PREFIX}{key}": value for key, value in encoded.items()}

reward = reward_model.compute_reward(batch)
```

`reward` is a tensor of shape `(batch_size,)`.

### Use the Reward Factory

You can also instantiate ROBOMETER through the reward factory:

```python
from lerobot.rewards import make_reward_model, make_reward_model_config, make_reward_pre_post_processors

cfg = make_reward_model_config(
    "robometer",
    pretrained_path="lerobot/Robometer-4B",
    device="cuda",
    image_key="observation.images.top",
)
reward_model = make_reward_model(cfg)
preprocessor, postprocessor = make_reward_pre_post_processors(cfg)
```

The preprocessor writes Qwen-VL tensors under the `observation.robometer.*` namespace, and `compute_reward()` reads those encoded tensors.

## Configuration Notes

### Backbone and Vocabulary

The published checkpoint uses a Qwen3-VL-4B backbone. ROBOMETER adds five special tokens to the tokenizer in a fixed order:

```text
<|split_token|>
<|reward_token|>
<|pref_token|>
<|sim_token|>
<|prog_token|>
```

`<|prog_token|>` is inserted after each frame and is the hidden-state position used for per-frame progress and success prediction. `<|split_token|>` and `<|pref_token|>` are used by the paper's pairwise trajectory preference objective. `<|reward_token|>` and `<|sim_token|>` are preserved for checkpoint compatibility.

The LeRobot config stores a serialized `vlm_config` with the post-resize vocabulary so the model can reload from `config.json` without downloading the base Qwen weights first. For `Qwen/Qwen3-VL-4B-Instruct`, the tokenizer length is `151669`, and the five ROBOMETER tokens produce the checkpoint vocabulary size `151674`.

### Progress Prediction

In the published checkpoint, progress is discrete. The progress head outputs logits over `progress_discrete_bins=10` uniformly spaced bin centers in `[0, 1]`. LeRobot converts these logits into a continuous value by applying a softmax and taking the expectation over bin centers, matching the upstream ROBOMETER implementation.

### Success Prediction

The success head outputs raw logits per frame. LeRobot converts them to probabilities with `sigmoid`. When `reward_output="success"`, `compute_reward()` thresholds the last-frame success probability using `success_threshold`.

## Limitations

- The current LeRobot integration is inference-only; it does not implement ROBOMETER training or preference-pair training.
- `compute_reward()` returns a scalar per sample for the LeRobot reward-model API, even though ROBOMETER predicts per-frame progress and success internally.
- ROBOMETER is video-language based; it does not use privileged robot state such as contact forces or object poses.

## References

- [ROBOMETER project](https://robometer.github.io/)
- [ROBOMETER paper](https://arxiv.org/abs/2603.02115)
- [Original ROBOMETER code](https://github.com/robometer/robometer)
- [Published ROBOMETER-4B checkpoint](https://huggingface.co/lerobot/Robometer-4B)
- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

## Citation

```bibtex
@inproceedings{liang2026robometer,
title = {Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons},
author={Anthony Liang and Yigit Korkmaz and Jiahui Zhang and Minyoung Hwang and Abrar Anwar and Sidhant Kaushik and Aditya Shah and Alex S. Huang and Luke Zettlemoyer and Dieter Fox and Yu Xiang and Anqi Li and Andreea Bobu and Abhishek Gupta and Stephen Tu and Erdem Biyik and Jesse Zhang},
year={2026},
booktitle={Robotics: Science and Systems 2026},
}
```

## License

This LeRobot integration follows the **Apache 2.0 License** used by LeRobot. Check the upstream ROBOMETER code and model pages for the licenses of the original implementation and released checkpoints.
