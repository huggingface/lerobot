# π₀.₅ (pi05)

This repository contains the Hugging Face port of **π₀.₅**, adapted from [OpenPI](https://github.com/Physical-Intelligence/openpi) by the Physical Intelligence.
It is designed as a **Vision-Language-Action model with open-world generalization**.

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

## Delta Actions (Relative Actions)

π₀.₅ supports training with **delta actions**, where the model learns relative offsets
from the current robot state instead of absolute joint positions. This mirrors the
`DeltaActions` transform from [OpenPI](https://github.com/Physical-Intelligence/openpi)
and can improve performance.

### How it works

1. **During preprocessing**, absolute actions are converted to deltas:
   `delta = action - state` (for selected joints).
2. The deltas are normalized using statistics computed from the delta distribution.
3. **During postprocessing**, predicted deltas are converted back to absolute:
   `absolute = delta + state`.

Joints listed in `delta_exclude_joints` (e.g., gripper) are kept absolute.

### Configuration

| Parameter              | Type        | Default       | Description                                                      |
| ---------------------- | ----------- | ------------- | ---------------------------------------------------------------- |
| `use_delta_actions`    | `bool`      | `False`       | Enable delta action training                                     |
| `delta_exclude_joints` | `list[str]` | `["gripper"]` | Joint names to keep absolute (matched by substring)              |
| `action_feature_names` | `list[str]` | `None`        | Auto-populated from dataset metadata at runtime by `make_policy` |

### Training example

```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=pi05 \
  --dataset.repo_id=your_org/your_dataset \
  --policy.use_delta_actions=true \
  --policy.delta_exclude_joints='["gripper"]'
```

When `use_delta_actions=true`, the training script automatically:

- Computes delta action statistics from the dataset (sampled chunk-level deltas)
- Replaces the standard action stats with delta stats for normalization
- Broadcasts these stats across all ranks in distributed training

---

## Citation

If you use this work, please cite both **OpenPI** and the π₀.₅ paper:

```bibtex
@misc{openpi2024,
  author       = {Physical Intelligence Lab},
  title        = {OpenPI: PyTorch Implementation of π0 and π0.5 Policies},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}},
  license      = {Apache-2.0}
}

@misc{intelligence2025pi05visionlanguageactionmodelopenworld,
  title        = {π₀.₅: a Vision-Language-Action Model with Open-World Generalization},
  author       = {Physical Intelligence and Kevin Black and Noah Brown and James Darpinian and Karan Dhabalia and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Manuel Y. Galliker and Dibya Ghosh and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Devin LeBlanc and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Allen Z. Ren and Lucy Xiaoyang Shi and Laura Smith and Jost Tobias Springenberg and Kyle Stachowicz and James Tanner and Quan Vuong and Homer Walke and Anna Walling and Haohuan Wang and Lili Yu and Ury Zhilinsky},
  year         = {2025},
  eprint       = {2504.16054},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2504.16054},
}
```

---

## License

This port follows the **Apache 2.0 License**, consistent with the original [OpenPI repository](https://github.com/Physical-Intelligence/openpi).
