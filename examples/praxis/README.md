# Praxis-1 examples

[`convert_checkpoint.py`](convert_checkpoint.py) turns a praxis checkpoint directory (`config.json`,
17 safetensors shards, `model.safetensors.index.json`, `SHA256SUMS`) into a LeRobot policy
directory: the LeRobot `config.json`, the shards (linked with `--link` instead of copying 54 GB)
and the two processor pipelines, with the canvas of the checkpoint (`video.tile_height`,
`video.antialias`) and the recommended sampler settings.

```bash
python examples/praxis/convert_checkpoint.py \
    --checkpoint /path/to/praxis-1-droid-m2-step50000 --out outputs/praxis-1-droid-m2 --link
```

See the [Praxis-1 guide](../../docs/source/praxis.mdx) for installation, the results in the praxis
simulation, fine-tuning and the parity checks.

## Fine-tuning helpers

[`select_droid_episodes.py`](select_droid_episodes.py) writes the DROID episodes of the praxis
recipe (an instruction, a training window plus the margin) as a JSON list for
`lerobot-train --dataset.episodes`:

```bash
python examples/praxis/select_droid_episodes.py --repo-id lerobot/droid_1.0.1 --revision refs/pr/11 \
    --out droid_episodes.json
```

[`merge_praxis_adapter.py`](merge_praxis_adapter.py) folds a LoRA adapter trained with
`lerobot-train --peft.*` into the weights and writes a plain policy directory:

```bash
python examples/praxis/merge_praxis_adapter.py --base runwayml/praxis-1-droid-lerobot \
    --adapter outputs/train/praxis_lora/checkpoints/last/pretrained_model --out outputs/praxis-1-droid-lora
```
