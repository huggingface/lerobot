# XAI Scripts — DaViT Vision Encoder Analysis

Explainability scripts for the XVLA fine-tuned model (`xvla-pouring-0.1`).  
Runs on a GPU server. Does **not** require the full LeRobot framework.

## Scripts

| File | Description |
|------|-------------|
| `load_model_test.py` | Smoke test — loads the model and prints DaViT feature-map shapes + VRAM stats |
| `xai_feature_maps.py` | Phase 1a — visualizes mean-channel activation maps at each DaViT stage |
| `xai_p0v_raw_attention.py` | Phase 1b — extracts language→image attention from the Florence-2 encoder (single image) |
| `xai_p0v_raw_attention_video.py` | Phase 1c — same as above, processed frame-by-frame over an MP4 video |
| `xai_utils.py` | Shared bootstrap: lerobot stubs, model loading, image preprocessing |
| `check_environment.py` | Verifies that all dependencies are installed correctly |

## Requirements

```
torch >= 2.0
safetensors
transformers
opencv-python
matplotlib
Pillow
numpy
```

Install on the server:

```bash
pip install safetensors transformers opencv-python matplotlib Pillow numpy
```

## Directory structure expected

By default the scripts resolve paths **relative to this `xai/` directory**:

```
<project_root>/
├── xai/                         ← this directory
│   └── *.py
├── XVLA original source/        ← XVLA_SOURCE_DIR (default)
│   ├── modeling_florence2.py
│   ├── modeling_xvla.py
│   └── ...
└── xvla-pouring-0.1/           ← XVLA_MODEL_DIR (default)
    ├── model.safetensors
    └── config.json
```

If your layout is different (e.g., scripts placed inside another repo), set the two environment variables below.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `XVLA_SOURCE_DIR` | `../XVLA original source` (relative to `xai/`) | Absolute path to the XVLA source directory containing `modeling_florence2.py` etc. |
| `XVLA_MODEL_DIR` | `../xvla-pouring-0.1` (relative to `xai/`) | Absolute path to the fine-tuned model directory containing `model.safetensors` and `config.json` |

### Setting env vars (Linux / bash)

```bash
export XVLA_SOURCE_DIR=/path/to/XVLA_original_source
export XVLA_MODEL_DIR=/path/to/xvla-pouring-0.1
```

Or inline for a single run:

```bash
XVLA_SOURCE_DIR=/data/xvla/src XVLA_MODEL_DIR=/data/xvla/model \
    python3 xai_feature_maps.py --image my_image.png
```

## Running the scripts

All commands should be run from the `xai/` directory.

### 1. Smoke test (verify model loads correctly)

```bash
python3 load_model_test.py
```

Expected output: DaViT stage shapes, VRAM usage, and `SMOKE TEST PASSED`.

### 2. Feature map visualization

```bash
python3 xai_feature_maps.py --image /path/to/image.png --alpha 0.55 --cmap turbo
```

Outputs saved to `xai/outputs/`:
- `feature_maps_grid.png` — 2×5 grid (raw + overlay per stage)
- `feature_maps_combined.png` — averaged heatmap across all stages
- `feature_maps_stage{0-3}.png` — individual per-stage overlays

### 3. Language→image attention (single image)

```bash
python3 xai_p0v_raw_attention.py \
    --image /path/to/image.png \
    --instruction "Pour coffee from the orange cup into the light blue cup."
```

Outputs saved to `xai/outputs/`:
- `p0v_per_layer_grid.png` — attention per encoder layer
- `p0v_head_analysis_layer11.png` — per-head analysis (last layer)
- `p0v_aggregated.png` — mean across all layers
- `p0v_overlay.png` — last-layer overlay

### 4. Language→image attention (video)

```bash
python3 xai_p0v_raw_attention_video.py \
    --video /path/to/video.mp4 \
    --instruction "Pour coffee from the orange cup into the light blue cup."
```

Output: `xai/outputs/p0v_video_<videoname>.mp4`

Use `--dry-run` on any script to verify model loading without running the full pipeline.

## Notes

- All scripts require a CUDA-capable GPU.
- The model checkpoint (`model.safetensors`) is **not** included in this repository. Obtain it separately and point `XVLA_MODEL_DIR` to its location.
- `xai_utils.py` bootstraps the lerobot dependency stubs at import time — it must not be run directly.
