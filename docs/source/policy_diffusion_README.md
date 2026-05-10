## Paper

https://diffusion-policy.cs.columbia.edu

## Language conditioning

Diffusion Policy keeps the original visuomotor behavior by default. For language-conditioned
multi-task datasets such as LIBERO, enable task-language conditioning with:

```bash
--policy.use_language_conditioning=true
```

When enabled, the policy preprocessor tokenizes the batch `task` field and the model appends a
projected CLIP text embedding to the Diffusion global conditioning vector. This requires
`transformers` in addition to the regular Diffusion dependencies. LIBERO users get this through the
`libero` extra; non-LIBERO users can install `diffusion` plus `transformers-dep`.

This flag changes the model architecture because the U-Net conditioning dimension grows. Existing
non-language Diffusion checkpoints remain compatible with the default disabled setting, but they
cannot be transparently evaluated as language-conditioned checkpoints without retraining or
fine-tuning with language conditioning enabled.

## Citation

```bibtex
@article{chi2024diffusionpolicy,
	author = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
	title ={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
	journal = {The International Journal of Robotics Research},
	year = {2024},
}
```
