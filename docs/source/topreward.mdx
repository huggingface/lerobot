# TOPReward

TOPReward is a **zero-shot reward model** that extracts token log-probabilities from an off-the-shelf vision-language model (VLM) as a robotic reward signal. Given a video trajectory and a task instruction, it returns the VLM's log-likelihood that the instruction is true — no fine-tuning required.

**Paper**: [TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics](https://arxiv.org/abs/2602.19313)
**Project**: [topreward.github.io](https://topreward.github.io/webpage/)
**Original code**: [github.com/TOPReward/TOPReward](https://github.com/TOPReward/TOPReward)
**Default backbone**: [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

## Overview

TOPReward asks a generic VLM how likely a task instruction is, **conditioned on the video** of a robot trying to complete that task. Concretely, given:

- A trajectory video (a sequence of frames).
- A task instruction (e.g. _"open the drawer"_).

it builds a chat prompt of the form

```text
<video>
"The above video shows a robot manipulation trajectory that completes the
 following task: <instruction> Decide whether the above statement is True
 or not. The answer is: True"
```

forwards it through the VLM, label-masks everything except the very last token, and reads back the log-probability of that token — by default the literal `"True"` that closes the suffix template. The resulting `log P("True" | video + prompt + instruction)` is the reward.

Because the method only depends on a frozen VLM, TOPReward is **zero-shot**: there are no fine-tuned weights to host. The "model" in LeRobot is a small wrapper around `transformers`' `Qwen3VLForConditionalGeneration` plus the label-masking logic. The processor owns the tokeniser and builds the full chat prompt (EO-1/Robometer pattern).

## What the LeRobot integration covers

- Standard `reward_model.type=topreward` configuration through LeRobot.
- VLM loading via the `transformers` `Qwen3VLForConditionalGeneration` API.
- Prompt assembly + tokenisation in the processor (matching upstream `QwenClient.compute_instruction_reward`).
- `compute_reward()` returns one scalar log-prob per sample.
- LeRobot reward-model save/load — `save_pretrained` writes only `config.json` (the VLM is identified by `vlm_name`).
- An offline labeling script that writes a `topreward_progress.parquet` (SARM-compatible schema) for RA-BC and overlay.

The current LeRobot port supports the **Qwen3-VL client only**. Other upstream clients (Gemini, OpenAI, Gemma, Molmo) can be added as follow-up extras.

## Installation Requirements

1. Install LeRobot following the [Installation Guide](./installation).
2. Install the TOPReward optional extra:

```bash
pip install -e ".[topreward]"
```

or, with `uv` from a source checkout:

```bash
uv sync --extra topreward
```

This pulls in `transformers`. The first time you run TOPReward, Hugging Face will also download the VLM weights from the Hub (~16 GB for Qwen3-VL-8B-Instruct). A GPU is strongly recommended.

## Model Inputs and Outputs

TOPReward expects:

- A trajectory video or sequence of frames.
- A natural-language task description.

In LeRobot datasets the preprocessor reads:

| Config field              | Default                     | Meaning                                       |
| ------------------------- | --------------------------- | --------------------------------------------- |
| `reward_model.image_key`  | `observation.images.top`    | Camera observation used by TOPReward          |
| `reward_model.task_key`   | `task`                      | Key in complementary data for the task string |
| `reward_model.max_frames` | `16`                        | Cap on frames per sample                      |
| `reward_model.fps`        | `2.0`                       | Metadata passed to the Qwen video processor   |
| `reward_model.vlm_name`   | `Qwen/Qwen3-VL-8B-Instruct` | Hugging Face Hub id of the underlying VLM     |

The model returns:

- `compute_reward(batch)`: one log-probability per sample. Higher = better task-video alignment. When `success_threshold` is finite, returns the binary thresholded value instead.

## Usage

### Load the reward model directly

```python
from lerobot.rewards.topreward import TOPRewardConfig, TOPRewardModel

cfg = TOPRewardConfig(
    vlm_name="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",
)
reward_model = TOPRewardModel(cfg)
```

### Use the reward factory

```python
from lerobot.rewards import make_reward_model, make_reward_model_config, make_reward_pre_post_processors

cfg = make_reward_model_config(
    "topreward",
    vlm_name="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",
    image_key="observation.images.top",
)
reward_model = make_reward_model(cfg)
preprocessor, postprocessor = make_reward_pre_post_processors(cfg)
```

The preprocessor tokenises the full prompt (video + prefix + instruction suffix), writes Qwen-VL tensors + `prompt_length` under `observation.topreward.*`. The model reads those tensors, label-masks based on `prompt_length`, and extracts the log-prob reward.

### Offline dataset labeling

Write a `topreward_progress.parquet` for RA-BC training and overlay videos:

```bash
# Sparse-dense (15 anchors per episode, matches upstream)
uv run python -m lerobot.rewards.topreward.compute_rabc_weights \
    --dataset-repo-id lerobot/libero_10_image \
    --num-samples 15 \
    --device cuda
```

Then render the progress overlay for any episode:

```bash
uv run examples/dataset/create_progress_videos.py \
    --repo-id lerobot/libero_10_image \
    --episode 0 \
    --progress-file topreward_progress.parquet \
    --gif
```

## Configuration Notes

### Prompt knobs

The default prompt mirrors the upstream paper:

```text
prompt_prefix = "The above video shows a robot manipulation trajectory that completes the following task: "
prompt_suffix_template = "{instruction} Decide whether the above statement is True or not. The answer is: True"
```

Both are exposed on `TOPRewardConfig` for ablation. The suffix template **must** contain `{instruction}`.

### Chat template

`add_chat_template=True` wraps the full prompt (including instruction) with the tokenizer's chat template before tokenisation. Default is `False`, matching the upstream paper's main experiments.

## Limitations

- The current LeRobot port is **inference-only and zero-shot**; `forward()` is not overridden and `is_trainable` returns `False`.
- Only the **Qwen3-VL family** is supported; other upstream clients are out of scope.
- TOPReward inherits the underlying VLM's biases.

## References

- [TOPReward project page](https://topreward.github.io/webpage/)
- [TOPReward paper](https://arxiv.org/abs/2602.19313)
- [Original TOPReward code](https://github.com/TOPReward/TOPReward)
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

## Citation

```bibtex
@article{chen2026topreward,
  title={TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics},
  author={Chen, Shirui and Harrison, Cole and Lee, Ying-Chun and Yang, Angela Jin and
          Ren, Zhongzheng and Ratliff, Lillian J and Duan, Jiafei and Fox, Dieter and
          Krishna, Ranjay},
  journal={arXiv preprint arXiv:2602.19313},
  year={2026}
}
```

## License

The original TOPReward codebase is MIT-licensed. The LeRobot port follows the LeRobot Apache 2.0 license; the wrapped Qwen3-VL weights are subject to the original Qwen license.
