# FLOWER

This repository contains the Hugging Face port of **FLOWER**, adapted from [FLOWER](https://github.com/intuitive-robots/flower_vla_calvin) by the intuitive-robots.
It is designed as a **Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies**.

---
## Training
To run the FLOWER pretrain:
```
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id='["${HF_USER}/my_dataset_1","${HF_USER}/my_dataset_2"]' \
  --dataset.collate_fn=lerobot.policies.flower.utils.FlowerDataCollator \
  --dataset.collate_fn_params='{"vlm_path": "microsoft/Florence-2-large"}' \
  --dataset.streaming=true \
  --policy.type=flower \
  --policy.device=cuda \
  --policy.training_stage=pretrain \
  --policy.vlm_path=microsoft/Florence-2-large \
  --policy.freeze_embeddings_only=true \
  --policy.load_pretrained=false \
  --batch_size=32 \
  --num_workers=4 \
```
To run the FLOWER SFT, you can use the pretrained model available at [mbreuss/flower_vla_pret](https://huggingface.co/mbreuss/flower_vla_pret). First, download the 360000_model_weights.pt file and specify its local path, then run:
```
accelerate launch \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=${HF_USER}/my_dataset \
  --dataset.collate_fn=lerobot.policies.flower.utils.FlowerDataCollator \
  --dataset.collate_fn_params='{"vlm_path": "microsoft/Florence-2-large"}' \
  --policy.type=flower \
  --policy.device=cuda \
  --policy.training_stage=sft \
  --policy.vlm_path=microsoft/Florence-2-large \
  --policy.freeze_embeddings_only=false \
  --policy.load_pretrained=true \
  --policy.pretrained_model_path='/path/to/your/360000_model_weights.pt' \
  --batch_size=32 \
  --num_workers=4 \
```
## Citation

If you use this work, please cite the **FLOWER** paper:

```bibtex
@inproceedings{
reuss2025flower,
title={{FLOWER}: Democratizing Generalist Robot Policies with Efficient Vision-Language-Flow Models},
author={Moritz Reuss and Hongyi Zhou and Marcel R{\"u}hle and {\"O}mer Erdin{\c{c}} Ya{\u{g}}murlu and Fabian Otto and Rudolf Lioutikov},
booktitle={9th Annual Conference on Robot Learning},
year={2025},
url={https://openreview.net/forum?id=JeppaebLRD}
}
```

---

## License

This port follows the **MIT License**, consistent with the original [FLOWER repository](https://github.com/intuitive-robots/flower_vla_calvin).