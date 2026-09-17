## Selecting the VLM backbone

`SmolVLAConfig.vlm_model_name` selects the vision-language backbone. It defaults to
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.

| Backbone | Params | Supported today | Notes |
|---|---|---|---|
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 500M | ✅ default | pretrained SmolVLA checkpoints available on the Hub |
| `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | 2.2B | ✅ (train from scratch) | more capacity; slower; no pretrained SmolVLA checkpoint |
| `google/paligemma-3b-pt-224` | 3B | ⚠️ not yet — see [#2104](https://github.com/huggingface/lerobot/issues/2104) | the action-expert wiring assumes the SmolVLM architecture |

**Caveats**

- **Switching the backbone requires training from scratch.** The encoder architecture and feature
  dimensions change, so you cannot fine-tune `smolvla-base` on a different backbone.
- Set `load_vlm_weights=True` only when initializing from pretrained SmolVLA weights; use `False`
  when training the expert from scratch.
- Larger backbones increase memory use and inference latency, which matters for real-time control
  on a physical arm (SO-101, etc.).

**Example**

```bash
lerobot-train --policy.type=smolvla \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --policy.load_vlm_weights=false
```

## Paper

https://arxiv.org/abs/2506.01844

## Citation

```bibtex
@article{shukor2025smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  author={Shukor, Mustafa and Aubakirova, Dana and Capuano, Francesco and Kooijmans, Pepijn and Palma, Steven and Zouitine, Adil and Aractingi, Michel and Pascal, Caroline and Russi, Martino and Marafioti, Andres and Alibert, Simon and Cord, Matthieu and Wolf, Thomas and Cadene, Remi},
  journal={arXiv preprint arXiv:2506.01844},
  year={2025}
}
```
