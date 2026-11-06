## Paper

https://diffusion-policy.cs.columbia.edu

## Training

The reference implementation maintains an exponential moving average (EMA) of the policy weights during training and evaluates the EMA weights. To reproduce this behavior, enable the trainer's EMA:

```bash
lerobot-train \
  --policy.type=diffusion \
  --ema.type=warmup \
  ...
```

Checkpoints then contain a directly loadable copy of the EMA weights next to the live ones, e.g. for evaluation:

```bash
lerobot-eval --policy.path=outputs/train/.../checkpoints/last/pretrained_model_ema ...
```

The `warmup` type (`--ema.inv_gamma`, `--ema.power`, ...) uses the reference implementation's values. Other types are a constant decay (`--ema.type=constant --ema.decay=0.99`, as in openpi's pi0/pi05 training) and the EDM2 power function (`--ema.type=power --ema.sigma_rel=0.1`).

## Citation

```bibtex
@article{chi2024diffusionpolicy,
	author = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
	title ={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
	journal = {The International Journal of Robotics Research},
	year = {2024},
}
```
