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

| Suite | Success rate | n_episodes |
| --- | ---: | ---: |
| libero_spatial | 97.6% | 500 |
| libero_object | 99.0% | 500 |
| libero_goal | 95.0% | 500 |
| libero_10 | 94.0% | 500 |
| **average** | **96.4%** | 2000 |

Reproduce: `lerobot-eval --policy.path=ZibinDong/fastwam_libero_uncond_2cam224 --policy.device=cuda --policy.torch_dtype=float32 --policy.n_action_steps=10 --env.type=libero --env.task=libero_spatial --env.observation_height=256 --env.observation_width=256 --eval.batch_size=1 --eval.n_episodes=50 --seed=0 --env.episode_length=300`.

For LIBERO-10, use `--env.task=libero_10 --env.episode_length=600`:

```bash
lerobot-eval \
    --policy.path=ZibinDong/fastwam_libero_uncond_2cam224 \
    --policy.device=cuda \
    --policy.torch_dtype=float32 \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_10 --env.observation_height=256 --env.observation_width=256 \
    --eval.batch_size=1 \
    --eval.n_episodes=50 \
    --seed=0 --env.episode_length=600
```
