## Paper

https://diffusion-policy.cs.columbia.edu

## Training

The reference implementation maintains an exponential moving average (EMA) of the policy weights during training and evaluates the EMA weights. To reproduce this behavior, enable the trainer's EMA shadow:

```bash
lerobot-train \
  --policy.type=diffusion \
  --ema.enable=true \
  ...
```

Checkpoints then contain a directly loadable copy of the EMA weights next to the live ones, e.g. for evaluation:

```bash
lerobot-eval --policy.path=outputs/train/.../checkpoints/last/pretrained_model_ema ...
```

The EMA decay schedule (`--ema.inv_gamma`, `--ema.power`, ...) defaults to the reference implementation's values. For a constant decay instead of the warmup schedule (e.g. to match openpi's pi0/pi05 training), set `--ema.decay=0.99`.

## Citation

```bibtex
@article{chi2024diffusionpolicy,
	author = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
	title ={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
	journal = {The International Journal of Robotics Research},
	year = {2024},
}
```
