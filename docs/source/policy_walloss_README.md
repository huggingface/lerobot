# WALL-OSS

This repository contains the Hugging Face port of [**WALL-OSS**](https://x2robot.com/en/research/68bc2cde8497d7f238dde690), a Vision-Language-Action model for cross-embodiment robotic control based on Qwen2.5-VL with flow matching/FAST action prediction.

---

## Model Overview

| Feature            | Description                                           |
| ------------------ | ----------------------------------------------------- |
| Base Model         | Qwen2.5-VL (Vision-Language Model)                    |
| Action Prediction  | Flow Matching (diffusion) or FAST (discrete tokens)   |
| Architecture       | Mixture of Experts (MoE) with action-specific routing |
| Multi-Modal Inputs | Vision (images/videos), Language, Proprioception      |

---

## Additional Resources

Paper: https://arxiv.org/pdf/2509.11766

Official Repository: https://github.com/X-Square-Robot/wall-x

Hugging Face: https://huggingface.co/x-square-robot

---

## Citation

If you use this work, please cite:

```bibtex
@article{zhai2025igniting,
    title   = {Igniting VLMs Toward the Embodied Space},
    author  = {Zhai, Andy and Liu, Brae and Fang, Bruno and Cai, Chalse and Ma, Ellie and Yin, Ethan and Wang, Hao and Zhou, Hugo and Wang, James and Shi, Lights and Liang, Lucy and Wang, Make and Wang, Qian and Gan, Roy and Yu, Ryan and Li, Shalfun and Liu, Starrick and Chen, Sylas and Chen, Vincent and Xu, Zach},
    journal = {arXiv preprint arXiv:2509.11766},
    year    = {2025}
}
```

---

## License

This model follows the **Apache 2.0 License**, consistent with the original [WallX repository](https://github.com/X-Square-Robot/wall-x).
