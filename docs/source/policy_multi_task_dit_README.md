# Multitask DiT Policy

## Training throughput

On CUDA, pass `--policy.compile_model=true`. The training step is dominated by the CLIP ViT-B/16
forward and backward, and `torch.compile` fuses the elementwise work around its GEMMs. Measured on one
A100 80GB in bf16 at an effective batch of 320: +22% to +31% samples/s and about 22% less peak memory
across LIBERO (256x256 images), RoboCasa (256x256 video, 3 cameras) and a 480x640 real-robot dataset.
It costs a one-off compile (~90s) at the first step and once per new batch shape.

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
