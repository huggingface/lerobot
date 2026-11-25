# SANA-Robot examples

[`convert_checkpoint.py`](convert_checkpoint.py) turns a released SANA-Robot checkpoint folder (one `.pth`
weights file, and `meta.json` and `normalization.json` when the checkpoint is finetuned on a dataset) into a
LeRobot policy directory: `config.json`, the weights as safetensors and the two processor pipelines, whose
normalizer holds the checkpoint's statistics.

```bash
python examples/sana_robot/convert_checkpoint.py \
    --checkpoint /path/to/SANA-Robot-AR-LIBERO --out outputs/SANA-Robot-AR-LIBERO-LeRobot
```

The defaults are those of the LIBERO checkpoint. For another checkpoint, pass the observation keys of its
cameras, their sizes, the spatial layout and the sampling settings (`--camera-keys`, `--view-sizes`,
`--spatial-layout`, `--n-action-steps`, `--num-inference-steps`, `--video-cfg-scale`).

See the [SANA-Robot guide](../../docs/source/sana_robot.mdx) for installation, inference and evaluation.
