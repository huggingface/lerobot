# Praxis-1 examples

[`convert_checkpoint.py`](convert_checkpoint.py) turns a praxis checkpoint directory (`config.json`,
17 safetensors shards, `model.safetensors.index.json`, `SHA256SUMS`) into a LeRobot policy
directory: the LeRobot `config.json`, the shards (linked with `--link` instead of copying 54 GB)
and the two processor pipelines, with the canvas of the checkpoint (`video.tile_height`,
`video.antialias`) and the recommended sampler settings. The directory gets its own `SHA256SUMS`. A
base checkpoint (`"embodiment": "base"`, no action heads) gets no processors; `init_heads.py` below
completes it.

```bash
python examples/praxis/convert_checkpoint.py \
    --checkpoint /path/to/praxis-1-droid --out outputs/praxis-1-droid --link
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

## Base checkpoint

[`compute_norm_stats.py`](compute_norm_stats.py) and [`init_heads.py`](init_heads.py) turn
`praxis-1-base` (the trunk without the head tensors of a rig) into the start of a fine-tune for a
single-arm rig: the normalization statistics of a LeRobot dataset over random training windows (no
video is read), then a policy directory with six new head tensors, the rig fields and the processors,
the base shards linked. See "Fine-tune the base checkpoint" in the guide.

```bash
python examples/praxis/compute_norm_stats.py --repo-id lerobot/droid_1.0.1 --revision refs/pr/11 \
    --state-key observation.state.cartesian --action-key action.cartesian \
    --episodes droid_episodes.json --n-windows 2000 --out stats.json
python examples/praxis/init_heads.py --base runwayml/praxis-1-base-lerobot --out outputs/praxis-1-myrig \
    --stats stats.json --tile-height 224 --tile-width 416 --antialias --link
```
