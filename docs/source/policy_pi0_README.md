# π₀ (pi0)

This repository contains the Hugging Face port of **π₀**, adapted from [OpenPI](https://github.com/Physical-Intelligence/openpi) by the Physical Intelligence.
It is designed as a **Vision-Language-Action model for general robot control**.

---

## Model Overview

| Feature              | π₀                                                     | π₀.₅                                      |
| -------------------- | ------------------------------------------------------ | ----------------------------------------- |
| Time Conditioning    | Concatenates time with actions via `action_time_mlp_*` | Uses `time_mlp_*` for AdaRMS conditioning |
| AdaRMS               | Not used                                               | Used in action expert                     |
| Tokenizer Length     | 48 tokens                                              | 200 tokens                                |
| Discrete State Input | False (Uses `state_proj` layer)                        | True                                      |
| Parameter Count      | Higher (includes state embedding)                      | Lower (no state embedding)                |

---

## Relative Actions

π₀ supports training with **relative actions**, where the model learns relative offsets
from the current robot state instead of absolute joint positions. This mirrors the
relative-action transform in OpenPI (`DeltaActions`) and can improve performance.

### How it works

1. **During preprocessing**, absolute actions are converted to relative offsets:
   `relative = action - state` (for selected joints).
2. The relative actions are normalized using statistics computed from the relative distribution.
3. **During postprocessing**, predicted relative actions are converted back to absolute:
   `absolute = relative + state`.

Joints listed in `relative_exclude_joints` (e.g., gripper) are kept absolute.

### Configuration

| Parameter                 | Type        | Default       | Description                                                      |
| ------------------------- | ----------- | ------------- | ---------------------------------------------------------------- |
| `use_relative_actions`    | `bool`      | `False`       | Enable relative-action training                                  |
| `relative_exclude_joints` | `list[str]` | `["gripper"]` | Joint names to keep absolute (matched by substring)              |
| `action_feature_names`    | `list[str]` | `None`        | Auto-populated from dataset metadata at runtime by `make_policy` |

### Training example

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=pi0 \
  --dataset.repo_id=your_org/your_dataset \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]'
```

When `use_relative_actions=true`, the training script automatically:

- Computes relative action statistics from the dataset (sampled chunk-level relative actions)
- Replaces the standard action stats with relative stats for normalization
- Broadcasts these stats across all ranks in distributed training

### Recomputing stats for an existing dataset

If you want to precompute relative action stats offline, use `recompute_stats` from
`lerobot.datasets`:

```python
from lerobot.datasets import LeRobotDataset, recompute_stats

dataset = LeRobotDataset("your_org/your_dataset")
dataset = recompute_stats(
    dataset,
    relative_action=True,
    relative_exclude_joints=["gripper"],
)
```

---

## Citation

If you use this work, please cite both **OpenPI** and the π₀ paper:

```bibtex
@misc{openpi2024,
  author       = {Physical Intelligence Lab},
  title        = {OpenPI: PyTorch Implementation of π0 and π0.5 Policies},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}},
  license      = {Apache-2.0}
}

@misc{black2024pi0visionlanguageactionflowmodel,
  title        = {π₀: A Vision-Language-Action Flow Model for General Robot Control},
  author       = {Kevin Black and Noah Brown and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Lucy Xiaoyang Shi and James Tanner and Quan Vuong and Anna Walling and Haohuan Wang and Ury Zhilinsky},
  year         = {2024},
  eprint       = {2410.24164},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2410.24164},
}
```

---

## License

This port follows the **Apache 2.0 License**, consistent with the original [OpenPI repository](https://github.com/Physical-Intelligence/openpi).
