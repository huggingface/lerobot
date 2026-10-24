## Research Paper

Paper: https://arxiv.org/abs/2603.16666

## Repository

Code: https://github.com/yuantianyuan01/FastWAM

Project page: https://yuantianyuan01.github.io/FastWAM/

## Citation

```bibtex
@article{yuan2026fastwam,
  title = {Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author = {Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal = {arXiv preprint arXiv:2603.16666},
  year = {2026},
  url = {https://arxiv.org/abs/2603.16666}
}
```

## Additional Resources

Base video model: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B

Released upstream checkpoints: https://huggingface.co/yuanty/fastwam

## Results

Evaluated on LIBERO with [`ZibinDong/fastwam_libero_uncond_2cam224`](https://huggingface.co/ZibinDong/fastwam_libero_uncond_2cam224):

| Suite          | Success rate | n_episodes |
| -------------- | -----------: | ---------: |
| libero_spatial |        97.6% |        500 |
| libero_object  |        99.0% |        500 |
| libero_goal    |        95.0% |        500 |
| libero_10      |        94.0% |        500 |
| **average**    |    **96.4%** |       2000 |

Reproduce: `lerobot-eval --policy.path=ZibinDong/fastwam_libero_uncond_2cam224 --policy.device=cuda --policy.dtype=float32 --policy.n_action_steps=10 --env.type=libero --env.task=libero_spatial --env.observation_height=256 --env.observation_width=256 --eval.batch_size=1 --eval.n_episodes=50 --seed=0 --env.episode_length=300`.

For LIBERO-10, use `--env.task=libero_10 --env.episode_length=600`:

```bash
lerobot-eval \
    --policy.path=ZibinDong/fastwam_libero_uncond_2cam224 \
    --policy.device=cuda \
    --policy.dtype=float32 \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_10 --env.observation_height=256 --env.observation_width=256 \
    --eval.batch_size=1 \
    --eval.n_episodes=50 \
    --seed=0 --env.episode_length=600
```

## Faster-WAM variant (arXiv 2608.02365)

Faster-WAM: Do World Action Models Need Deep Action Modules?
(https://arxiv.org/abs/2608.02365, Huawei Noah's Ark Montreal / McGill / Mila / U Toronto)
is a world-model-centric upgrade of the Fast-WAM architecture above. It is a DIFFERENT
paper from Fast-WAM (arXiv 2603.16666, this policy's default) and from the unrelated
Faster-WAM (arXiv 2608.04404, hustvl). Always cite the arXiv ID, never the name alone.

Three opt-in config flags on `FastWAMConfig` implement it (all default off, which keeps
the Fast-WAM behavior and checkpoint layout exactly):

- `action_expert_layers=1`: a single-layer action expert (paper Sec. 3.1; about 28.6M
  parameters with hidden 1024, FFN 4096, QKV to the video backbone's 3072 attention
  width). Any depth between 1 and the video backbone depth works.
- `lite_kv_fusion=True`: Dock of Transformers Lite KV-Fusion (paper Eq. 1). Each action
  layer learns, per attention head, a softmax mixture over the native K/V cached across
  ALL video layers (fused after the backbone's QK-Norm and RoPE). Adds
  `num_heads x L_v x L_a` parameters (720 for the paper's 24x30x1). Required whenever
  `action_expert_layers` differs from the video depth.
- `retracted_1d_rope=True`: retracted 1D-RoPE (paper Eq. 3). The fused video keys are
  de-rotated with the backbone's native 3D rotary transform and re-encoded on the action
  queries' 1D temporal axis (identity for the observed frame at t=0). Requires
  `lite_kv_fusion`. Optional `world_model_only_conditioning=True` additionally builds the
  action expert without text cross-attention and never passes it text or proprioceptive
  context (paper Sec. 2.2); language and proprioception condition only the video backbone.

The paper's full configuration is
`action_expert_layers=1, lite_kv_fusion=True, retracted_1d_rope=True,
world_model_only_conditioning=True`. Training uses the same joint video + action
flow-matching loss as Fast-WAM; gradients from the action loss reach the video backbone
through the fusion. At inference the video KV cache is prefilled once per chunk and the
fused context is fixed across action denoising steps, so repeated denoising only runs the
shallow expert. No trained Faster-WAM checkpoints are released here. You can initialize
from a default FastWAM backbone and then fine-tune on your own data:

```bash
lerobot-train --policy.path=lerobot/fastwam_base \
  --policy.action_expert_layers=1 --policy.lite_kv_fusion=true \
  --policy.retracted_1d_rope=true --policy.world_model_only_conditioning=true \
  --dataset.repo_id=YOUR_DATASET --policy.push_to_hub=false
```

This retains compatible backbone weights and the leading action layers. It discards
unused action layers and text modules, and initializes fusion logits to zero. This
conversion requires fine-tuning; it does not preserve the trained policy's behavior.
Keep the video backbone depth unchanged.
The Python loader permits it with `strict=False` (the default). When loading an already
converted Faster-WAM checkpoint, keep its saved Faster-WAM options and action depth.

The implementation is derived from the paper and the Apache-2.0 LeRobot fastwam code.
The public repository [TeamFasterwam/FasterWAM](https://github.com/TeamFasterwam/FasterWAM)
licenses its original contributions under the
[PolyForm Noncommercial License 1.0.0](https://github.com/TeamFasterwam/FasterWAM/blob/main/LICENSE).
Its code was not copied, translated, or adapted.
