## 研究论文

论文：https://arxiv.org/abs/2603.16666

## 仓库

代码：https://github.com/yuantianyuan01/FastWAM

项目页面：https://yuantianyuan01.github.io/FastWAM/

## 引用

```bibtex
@article{yuan2026fastwam,
  title = {Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author = {Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal = {arXiv preprint arXiv:2603.16666},
  year = {2026},
  url = {https://arxiv.org/abs/2603.16666}
}
```

## 其他资源

基础视频模型：https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B

已发布的上游检查点：https://huggingface.co/yuanty/fastwam

## 结果

在 LIBERO 上使用 [`ZibinDong/fastwam_libero_uncond_2cam224`](https://huggingface.co/ZibinDong/fastwam_libero_uncond_2cam224) 评估：

| 套件 | 成功率 | 回合数 |
| -------------- | -----------: | ---------: |
| libero_spatial | 97.6% | 500 |
| libero_object | 99.0% | 500 |
| libero_goal | 95.0% | 500 |
| libero_10 | 94.0% | 500 |
| **平均** | **96.4%** | 2000 |

复现命令：`lerobot-eval --policy.path=ZibinDong/fastwam_libero_uncond_2cam224 --policy.device=cuda --policy.torch_dtype=float32 --policy.n_action_steps=10 --env.type=libero --env.task=libero_spatial --env.observation_height=256 --env.observation_width=256 --eval.batch_size=1 --eval.n_episodes=50 --seed=0 --env.episode_length=300`。

对于 LIBERO-10，使用 `--env.task=libero_10 --env.episode_length=600`：

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