# MolmoAct2

This repository contains the LeRobot policy implementation of
[MolmoAct2](https://allenai.org/blog/molmoact2), ported into LeRobot for
training, evaluation, checkpointing, and dataset compatibility.

This implementation currently supports training and evaluation for the regular
MolmoAct2 model. MolmoAct2-Think, which supports adaptive depth reasoning, is
not included in this LeRobot policy yet and is coming soon.

For the original MolmoAct2 training code used for the experiments reported in
the paper, see [allenai/molmoact2](https://github.com/allenai/molmoact2).

## LIBERO Evaluation

Important: we found that `num_steps_wait=10` does not reliably let the LIBERO
scene stabilize and can degrade measured success. All LIBERO evaluation results
reported for this LeRobot implementation use `num_steps_wait=50`.

## Citation

```bibtex
@misc{fang2026molmoact2actionreasoningmodels,
      title={MolmoAct2: Action Reasoning Models for Real-world Deployment},
      author={Haoquan Fang and Jiafei Duan and Donovan Clay and Sam Wang and Shuo Liu and Weikai Huang and Xiang Fan and Wei-Chuan Tsai and Shirui Chen and Yi Ru Wang and Shanli Xing and Jaemin Cho and Jae Sung Park and Ainaz Eftekhar and Peter Sushko and Karen Farley and Angad Wadhwa and Cole Harrison and Winson Han and Ying-Chun Lee and Eli VanderBilt and Rose Hendrix and Suveen Ellawela and Lucas Ngoo and Joyce Chai and Zhongzheng Ren and Ali Farhadi and Dieter Fox and Ranjay Krishna},
      year={2026},
      eprint={2605.02881},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.02881},
}
```

## License

This model is licensed under Apache 2.0. It is intended for research and
educational use in accordance with
[Ai2's Responsible Use Guidelines](https://allenai.org/responsible-use),
consistent with [allenai/molmoact2](https://github.com/allenai/molmoact2).
