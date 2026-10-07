# 实时分块（RTC）

本模块包含了 LeRobot 对**实时分块（RTC）**的实现，这是一种针对基于流匹配的策略的推理时技术。

**注意**：RTC 本身不是一种策略，而是一种推理增强技术，可与基于流匹配的策略配合使用，包括 [π₀](../pi0/)、[π₀.₅](../pi05/) 和 [SmolVLA](../smolvla/)。

---

## 引用

如果您在研究中使用了实时分块，请引用：

```bibtex
@misc{openpi2024,
  author       = {Physical Intelligence Lab},
  title        = {OpenPI: PyTorch Implementation of π0 and π0.5 Policies},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}},
  license      = {Apache-2.0}
}

@misc{black2025realtimeexecutionactionchunking,
      title={Real-Time Execution of Action Chunking Flow Policies},
      author={Kevin Black and Manuel Y. Galliker and Sergey Levine},
      year={2025},
      eprint={2506.07339},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.07339},
}
```

---

## 许可证

本实现遵循 **Apache 2.0 许可证**，与 LeRobot 项目保持一致。