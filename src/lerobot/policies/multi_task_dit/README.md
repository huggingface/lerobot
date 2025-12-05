# Multi-Task DiT Policy

For details describing the architecture, see the citations and the blog post from Bryson Jones: https://brysonkjones.substack.com/p/dissecting-multi-task-diffusion-transformer-policy

## Trainining and Inference Baseline Recommendations:

### Training

- Number of demonstrations: >100 per task
- Batch Size: 320
- Objective: Diffusion
- Cameras: At least two, with one egocentric view per arm

### Inference

- GPU: 5070 Ti or above in performance
- Sampling:
  - Strategy: DDIM
  - Number of Timesteps: 10

## Citation

If you use this work, please cite the following works:

```bibtex
@misc{jones2025multitaskditpolicy,
  author = {Bryson Jones},
  title = {Dissecting Multitask Diffusion Transformer Policy},
  year = {2025},
  url = {https://brysonkjones.substack.com/p/dissecting-multitask-diffusion-transformer-policy},
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
