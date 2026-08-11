# DM0.5 for LeRobot

DM05, also released as DM0.5, is Dexmal's Vision-Language-Action model for open-world robot control. This adapter
provides standard LeRobot training, evaluation, checkpointing, and Hub loading.

See the [complete user guide](./dm05.mdx) for dataset and normalization details.

## Installation

```bash
pip install -e ".[training,dm05]"   # training
pip install -e ".[libero,dm05]"     # LIBERO evaluation on Linux
```

## Checkpoint loading

`Dexmal/DM05` is the raw OpenDM checkpoint. Use the self-contained `Dexmal/DM05-Lerobot` checkpoint with
`DM05Policy.from_pretrained()` or `--policy.path`.

## Training

```bash
lerobot-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.video_backend=pyav \
  --policy.path=Dexmal/DM05-Lerobot \
  --policy.add_state=false \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.repo_id=your_repo_id \
  --output_dir=outputs/train/dm05-libero \
  --steps=50000 \
  --batch_size=8 \
  --policy.device=cuda
```

For local-only training, replace `--policy.repo_id=...` with `--policy.push_to_hub=false`.

This LIBERO recipe matches OpenDM: `policy.add_state=false` and stored actions are learned unchanged. Keep the
checkpoint default `policy.add_state=true` unless another recipe specifies otherwise. Relative targets require
matching state/action dimensions and matching relative-action statistics; environment control mode is separate.

DM05 expects complete state/action statistics in `meta/stats.json`. Review the
[normalization guide](./dm05.mdx#normalization-statistics) before training a non-standard dataset or using relative
actions.

## Evaluation

Evaluate a fine-tuned checkpoint:

```bash
MUJOCO_GL=egl lerobot-eval \
  --policy.path=/path/to/checkpoint/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.control_mode=relative \
  --policy.device=cuda
```

Use the complete checkpoint directory; policy config, tokenizer, preprocessing state, and weights are all required.

## Resources

- [DM0.5 technical blog](https://www.dexmal.com/blog/dm0.5/index_en.html)
- [OpenDM repository](https://github.com/dexmal/OpenDM)
- [Raw DM0.5 model](https://huggingface.co/Dexmal/DM05)

## Citation

```bibtex
@misc{dm05,
    title  = {{DM0.5}: An Open-World Foundation Model for General-Purpose Embodied Intelligence},
    author = {{Dexmal Team}},
    month  = {July},
    year   = {2026},
    url    = {https://www.dexmal.com/blog/dm0.5/index_en.html}
}
```

The source code is licensed under Apache-2.0. Model weights follow the license attached to their model card.
