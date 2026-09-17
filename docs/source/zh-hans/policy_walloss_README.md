# WALL-OSS

本仓库包含 Hugging Face 对 [**WALL-OSS**](https://x2robot.com/en/research/68bc2cde8497d7f238dde690) 的移植，这是一个基于 Qwen2.5-VL 的跨实体机器人控制视觉-语言-动作模型，采用流匹配/FAST 动作预测。

---

## 模型概览

| 特性 | 描述 |
| ------------------ | ----------------------------------------------------- |
| 基础模型 | Qwen2.5-VL（视觉-语言模型） |
| 动作预测 | 流匹配（扩散）或 FAST（离散令牌） |
| 架构 | 混合专家（MoE），带动作特定路由 |
| 多模态输入 | 视觉（图像/视频）、语言、本体感知 |

---

## 其他资源

论文：https://arxiv.org/pdf/2509.11766

官方仓库：https://github.com/X-Square-Robot/wall-x

Hugging Face：https://huggingface.co/x-square-robot

---

## 引用

如果您使用此工作，请引用：

```bibtex
@article{zhai2025igniting,
    title   = {Igniting VLMs Toward the Embodied Space},
    author  = {Zhai, Andy and Liu, Brae and Fang, Bruno and Cai, Chalse and Ma, Ellie and Yin, Ethan and Wang, Hao and Zhou, Hugo and Wang, James and Shi, Lights and Liang, Lucy and Wang, Make and Wang, Qian and Gan, Roy and Yu, Ryan and Li, Shalfun and Liu, Starrick and Chen, Sylas and Chen, Vincent and Xu, Zach},
    journal = {arXiv preprint arXiv:2509.11766},
    year    = {2025}
}
```

---

## 许可证

本模型遵循 **Apache 2.0 许可证**，与原始 [WallX 仓库](https://github.com/X-Square-Robot/wall-x) 保持一致。