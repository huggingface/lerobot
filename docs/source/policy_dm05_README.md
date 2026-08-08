# DM05

This repository contains the LeRobot policy implementation of DM05, also
released as DM0.5, Dexmal's Vision-Language-Action model for open-world robot
control.

The integration supports training, evaluation, checkpointing, and Hub-style
loading through the standard LeRobot policy interfaces:

- `policy.type=dm05`
- `DM05Policy.from_pretrained()`
- `lerobot-train`
- `lerobot-eval`

For the user-facing LeRobot guide, see `docs/source/dm05.mdx`.

---

## Model Overview

| Feature           | Description                                                       |
| ----------------- | ----------------------------------------------------------------- |
| Policy type       | `dm05`                                                            |
| Inputs            | Vision observations, robot state, and language task text          |
| Outputs           | Continuous action chunks                                          |
| Backbone          | Gemma3 4B VLM with a 680M Action Expert                           |
| Action generation | Flow Matching                                                     |
| Checkpoint format | Self-contained LeRobot checkpoint with DM05 core assets           |
| Processor assets  | Transformers tokenizer/processor files plus `chat_template.jinja` |
| Action adaptation | Uses raw LeRobot state/action dims with model-side pad/mask       |
| Normalization     | Uses DM05 model-side normalization                                |

---

## Installation

Install the DM05 extra together with the dependencies needed for the workflow you
are running:

```bash
pip install -e ".[dm05]"
pip install -e ".[training,dm05]"   # training / fine-tuning
pip install -e ".[libero,dm05]"     # LIBERO simulation eval on Linux
```

For LIBERO rollouts, set MuJoCo to EGL rendering in headless jobs:

```bash
export MUJOCO_GL=egl
```

---

## Checkpoints

Every training command should set `--policy.pretrained_name_or_path` explicitly.
Use `Dexmal/DM05` for the public base model, or replace it with a local
self-contained `pretrained_model` directory when continuing from another DM05
checkpoint.

Self-contained LeRobot checkpoints should include:

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
norm_stats.json
tokenizer.json
tokenizer_config.json
processor_config.json
chat_template.jinja
```

`chat_template.jinja` is required by the Transformers processor for prompt
rendering and is included in the DM05-specific Hub upload allow list.

---

## Training / Fine-Tuning

A standard LeRobot training command sets both `policy.type=dm05` and the base
checkpoint explicitly. Use `Dexmal/DM05` for the public base model, or replace
that value with another DM05 core checkpoint or local self-contained
`pretrained_model` directory:

```bash
lerobot-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.video_backend=pyav \
  --policy.type=dm05 \
  --policy.pretrained_name_or_path=Dexmal/DM05 \
  --policy.action_mode=absolute \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --output_dir=outputs/train/dm05-libero \
  --steps=50000 \
  --batch_size=8 \
  --policy.device=cuda
```

`--policy.path` loads a complete saved LeRobot policy configuration. Keep it for
cases where you intentionally want to load that saved policy config rather than
starting a new SFT run from `policy.type=dm05` plus `pretrained_name_or_path`.

Common LIBERO settings:

| Parameter                                     | Typical value                     | Description                                      |
| --------------------------------------------- | --------------------------------- | ------------------------------------------------ |
| `policy.pretrained_name_or_path`              | `Dexmal/DM05` or local checkpoint | Base DM05 weights and colocated processor assets |
| `policy.action_mode`                          | `absolute`                        | Learn dataset actions directly                   |
| `policy.chunk_size` / `policy.n_action_steps` | `10` / `10`                       | LIBERO action chunk length                       |
| `policy.use_absolute_action`                  | `false`                           | Only set true when converting relative outputs   |

---

## Evaluation

Evaluate a saved LeRobot checkpoint with `policy.path`:

```bash
MUJOCO_GL=egl lerobot-eval \
  --policy.path=/path/to/checkpoint/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --policy.device=cuda
```

Increase evaluation scope by changing the suite, task ids, and episode count as
needed for the benchmark you are running.

---

## Data Format Notes

DM05 expects standard LeRobot datasets with image observations,
`observation.state`, `action`, and language task descriptions. If several visual
keys are present and the camera order matters, set `policy.image_keys` to the
ordered list of `observation.images.*` keys.

`action_mode=absolute` trains on the dataset `action` field directly.
`action_mode=relative` trains on state-relative offsets; set
`use_absolute_action=true` only when inference must convert relative outputs back
to absolute actions.

---

## Additional Resources

- LeRobot DM05 guide: `docs/source/dm05.mdx`
- DM0.5 technical blog: <https://www.dexmal.com/blog/dm0.5/index_en.html>
- OpenDM repository: <https://github.com/dexmal/OpenDM>
- Public model release: <https://huggingface.co/Dexmal/DM05>

---

## Citation

Use the citation and attribution guidance published with the DM0.5 technical
blog or the `Dexmal/DM05` model card.

---

## License

The DM05 LeRobot source code in this repository is licensed under Apache-2.0.
DM05 model weights and checkpoints follow the license terms attached to the
corresponding DM05 model card or release. Confirm those terms before publishing
fine-tuned DM05 checkpoints to the Hugging Face Hub.
