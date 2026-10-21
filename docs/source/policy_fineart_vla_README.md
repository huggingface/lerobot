# FineART-VLA

FineART-VLA is a single end-to-end Vision-Language-Action policy from Scale AI and
Hugging Face. It extends π₀.₅ with a trained language head that predicts the next
subtask in language (System 2), which conditions a flow-matching action expert
(System 1). Knowledge insulation keeps the continuous-action gradients out of the
pretrained VLM, and optional FAST action tokens add discrete action supervision.

The policy is mid-trained on FineART, a bimanual manipulation dataset of 40,543
episodes (1,718 hours) with 533,913 subtask annotations. See the
[FineART-VLA guide](https://huggingface.co/docs/lerobot/fineart_vla) for training
and inference.

```bibtex
@misc{choghari2026fineart,
  title  = {FineART: Fine-Grained Annotated Robotic Trajectory Dataset and Vision-Language-Action Model for Bimanual Manipulation},
  author = {Choghari, Jade and Kooijmans, Pepijn and Agarwal, Mansi and Ciftci, Yusuf Umut and Doriwala, Aseem and Weaver, Catherine and Sivapurapu, Mouli and Yang, Kai and Lee, Jackson and Wolf, Thomas and Mannam, Pragna},
  year   = {2026},
}
```
