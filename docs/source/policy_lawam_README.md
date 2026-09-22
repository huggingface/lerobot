## Paper

https://arxiv.org/abs/2606.15768

## Project

Project page: https://rlinf.github.io/LaWAM/

Code: https://github.com/RLinf/LaWAM

Released checkpoints: https://huggingface.co/collections/jialei02/lawam-checkpoints

## LIBERO Benchmark

Released LIBERO SFT checkpoint, 50 episodes per task, using MuJoCo 3.3.2:

LIBERO benchmark success rates are sensitive to the MuJoCo simulator version. Newer MuJoCo releases can materially reduce measured LIBERO success rates, including for policies beyond LaWAM. This is an evaluation-environment compatibility issue rather than a LaWAM adapter issue; use `mujoco==3.3.2` to reproduce the released LaWAM LIBERO and paper results exactly.

| Suite            | Successes | Episodes | Success rate |
| ---------------- | --------: | -------: | -----------: |
| `libero_spatial` |       492 |      500 |        98.4% |
| `libero_object`  |       498 |      500 |        99.6% |
| `libero_goal`    |       490 |      500 |        98.0% |
| `libero_10`      |       490 |      500 |        98.0% |
| `overall`        |      1970 |     2000 |        98.5% |

## Citation

```bibtex
@misc{chen2026lawam,
  title = {LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies},
  author = {Chen, Jialei and Wang, Kai and Chen, Kang and Chen, Shuaihang and Gao, Feng and Tang, Wenhao and Li, Zhiyuan and Liu, Weilin and Yao, Zhuyu and Li, Boxun and Xu, Yuanbo and Yu, Chao},
  journal = {arXiv preprint arXiv:2606.15768},
  year = {2026},
  archiveprefix = {arXiv},
  primaryclass = {cs.RO},
}
```
