# Cosmos3

Cosmos3 is a world action model for robot control, built on NVIDIA's Cosmos 3 omnimodal world models. From the camera observations and a language instruction it jointly predicts the future video frames and the next chunk of robot actions. Two sizes are available: Cosmos3 Nano (`--policy.type=cosmos3_nano`) and Cosmos3 Edge (`--policy.type=cosmos3_edge`).

See the [Cosmos3 documentation](https://huggingface.co/docs/lerobot/cosmos3) for the full training and evaluation guide.

## Installation

```bash
pip install -e ".[cosmos3,training]"
```

## Training

Fine-tune Cosmos3 Edge from the base checkpoint on LIBERO-10 with one GPU:

```bash
lerobot-train \
  --policy.type=cosmos3_edge \
  --policy.pretrained_path=lerobot/cosmos3-edge \
  --policy.load_action_modules=false \
  --policy.gradient_checkpointing=true \
  --policy.repo_id=${HF_USER}/cosmos3_edge_libero_10 \
  --dataset.repo_id=lerobot/libero_10-cosmos3 \
  --accelerator.mixed_precision=bf16 \
  --output_dir=outputs/train/cosmos3_edge_libero_10
```

Cosmos3 Nano trains with FSDP2 on several GPUs. The documentation gives the multi GPU commands of the native LIBERO and DROID recipes.

## Resources

- Technical report: https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf
- Code: https://github.com/NVIDIA/cosmos-framework
- Models: https://huggingface.co/collections/nvidia/cosmos3

## Citation

```bibtex
@techreport{nvidia2026cosmos3,
  title  = {Cosmos 3: Omnimodal World Models for physical AI},
  author = {{NVIDIA Cosmos Team}},
  year   = {2026},
  url    = {https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf}
}
```

## License

The LeRobot implementation follows the Apache 2.0 License of LeRobot. The Cosmos3 weights are released under the [OpenMDW-1.1 License](https://openmdw.ai/license/1-1/).
