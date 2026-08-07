# VLA-JEPA

本仓库包含 LeRobot 对 **VLA-JEPA** 的移植，这是一个结合了 Qwen3-VL 语言骨干网络、自监督视频世界模型（V-JEPA2）和流匹配 DiT 动作头的视觉-语言-动作模型。

从 [ginwind/VLA-JEPA](https://huggingface.co/ginwind/VLA-JEPA) 转换而来。

---

## 架构概览

| 组件 | 模块 | 角色 |
| ----------------------- | --------------------------------- | ------------------------------------------------------- |
| **Qwen3-VL 骨干网络** | `Qwen3VLInterface` | 将图像和语言指令融合为上下文令牌 |
| **DiT-B 动作头** | `VLAJEPAActionHead` | 对动作块进行流匹配扩散 |
| **V-JEPA2 世界模型** | `ActionConditionedVideoPredictor` | 自监督视频预测损失（仅训练时） |

推理时仅使用 Qwen 骨干网络和动作头；世界模型不需要。

---

## 引用

```bibtex
@misc{sun2026vlajepaenhancingvisionlanguageactionmodel,
  title         = {VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model},
  author        = {Jingwen Sun and Wenyao Zhang and Zekun Qi and Shaojie Ren and Zezhi Liu and Hanxin Zhu and Guangzhong Sun and Xin Jin and Zhibo Chen},
  year          = {2026},
  eprint        = {2602.10098},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2602.10098},
}
```

---

## 许可证

权重根据原始 [ginwind/VLA-JEPA](https://huggingface.co/ginwind/VLA-JEPA) 仓库的许可条款分发（**Apache 2.0 许可证**）。LeRobot 集成代码遵循 **Apache 2.0 许可证**。