# Multitask DiT Policy

## Encoders

`vision_encoder_name` and `text_encoder_name` each accept multiple encoder families (any HuggingFace
model id; the loader is chosen from the checkpoint's `model_type`). CLIP is the default and unchanged.

| Family         | Vision | Text            | Global feature                      |
| -------------- | ------ | --------------- | ----------------------------------- |
| CLIP (default) | ✅     | ✅              | CLS token                           |
| DINOv3         | ✅     | — (vision-only) | CLS token (register tokens skipped) |
| SigLIP 2       | ✅     | ✅              | attention-pooled `pooler_output`    |

Use a **fixed-resolution** SigLIP 2 checkpoint (e.g. `google/siglip2-base-patch16-224`); NaFlex variants
are not supported. SigLIP 2 text uses a 64-token context, so set `tokenizer_max_length=64`. The tokenizer
is selected automatically from `text_encoder_name`. DINOv3/SigLIP 2 diverge from the CLIP baseline of the
original blog/paper; CLIP remains the default. The three above are what we test and recommend, but other
CLIP/DINO/SigLIP checkpoints (any size) work through the same paths — broadly, any vision encoder with a
CLS token (or SigLIP-style pooled output) and any text encoder with a pooled output. See
[`multi_task_dit.mdx`](./multi_task_dit.mdx) for configuration examples.

## Citation

If you use this work, please cite the following works:

```bibtex
@misc{jones2025multitaskditpolicy,
  author = {Bryson Jones},
  title = {Dissecting and Open-Sourcing Multitask Diffusion Transformer Policy},
  year = {2025},
  url = {https://brysonkjones.substack.com/p/dissecting-and-open-sourcing-multitask-diffusion-transformer-policy},
  note = {Blog post}
}
```

```bibtex
@misc{trilbmteam2025carefulexaminationlargebehaviormodels,
  author       = {TRI LBM Team},
  title        = {A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation},
  year         = {2025},
  eprint       = {arXiv:2507.05331},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url          = {https://arxiv.org/abs/2507.05331}
}
```

```bibtex
@misc{bostondynamics2025largebehaviormodelsatlas,
  author       = {Boston Dynamics and TRI Research Team},
  title        = {Large Behavior Models and Atlas Find New Footing},
  year         = {2025},
  url          = {https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/},
  note         = {Blog post}
}
```
