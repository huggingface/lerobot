# VLA-JEPA

This repository contains the LeRobot port of **VLA-JEPA**, a Vision-Language-Action model that combines a Qwen3-VL language backbone with a self-supervised video world model (V-JEPA2) and a flow-matching DiT action head.

Converted from [ginwind/VLA-JEPA](https://huggingface.co/ginwind/VLA-JEPA).

---

## Architecture Overview

| Component               | Module                            | Role                                                    |
| ----------------------- | --------------------------------- | ------------------------------------------------------- |
| **Qwen3-VL backbone**   | `Qwen3VLInterface`                | Fuses images + language instruction into context tokens |
| **DiT-B action head**   | `VLAJEPAActionHead`               | Flow-matching diffusion over the action chunk           |
| **V-JEPA2 world model** | `ActionConditionedVideoPredictor` | Self-supervised video prediction loss (training only)   |

At inference time only the Qwen backbone and action head are used; the world model is not needed.

---

## Citation

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

## License

Weights are distributed under the license terms of the original [ginwind/VLA-JEPA](https://huggingface.co/ginwind/VLA-JEPA) repository (**Apache 2.0 License**). The LeRobot integration code follows the **Apache 2.0 License**.
