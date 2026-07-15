# TLabel Tactile Data → LeRobot Conversion

This example demonstrates how to convert [TLabel](https://github.com/liesliy/tlabel) tactile sensor datasets into [LeRobotDataset](https://github.com/huggingface/lerobot) v3.0 format.

## Why TLabel?

[TLabel](https://github.com/liesliy/tlabel) (PyPI: `tlabel`) is a sensor-agnostic tactile data annotation toolkit that provides:

- **Unified 22-dimension annotation schema** (TLabel Format v2)
- Support for multiple tactile sensors: GelSight, DIGIT, PaXini, Daimon, Touchd, UniVTAC, VTAC
- Standardized data export (JSON/CSV)
- Quality grading and event-based annotation

## Installation

```bash
pip install tlabel lerobot
```

## Quick Start

```bash
python convert_tlabel_to_lerobot.py \
    --input-dir /path/to/tlabel_dataset/ \
    --repo-id username/tactile_dataset \
    --fps 30 \
    --sensor-type gelsight
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--input-dir` | Path to TLabel exported dataset | Required |
| `--repo-id` | LeRobot dataset repo ID (e.g. `user/dataset`) | Required |
| `--fps` | Sampling rate in Hz | 30 |
| `--sensor-type` | Sensor type: `gelsight`, `paxini`, `daimon`, `touchd`, `univtac`, `vtac` | `gelsight` |
| `--output-dir` | Local output directory (instead of pushing to Hub) | `./lerobot_data/` |
| `--push-to-hub` | Push dataset to Hugging Face Hub | Flag |
| `--task` | Task description string | `"tactile manipulation"` |
| `--config` | Path to custom feature config YAML | None (uses defaults) |

## Feature Mapping

TLabel's unified 22-dim schema maps to LeRobot features as follows:

| TLabel Dimension | LeRobot Feature Key | Shape |
|---|---|---|
| `contact` | `observation.tactile.contact` | (1,) |
| `force_magnitude`, `force_direction`, `force_peak` | `observation.tactile.force` | (3,) |
| `deformation_magnitude`, `temporal_deformation_rate` | `observation.tactile.deformation` | (2,) |
| `slip_entropy`, `slip_event` | `observation.tactile.slip` | (2,) |
| `texture_energy` | `observation.tactile.texture` | (1,) |
| `contact_area`, `centroid_x`, `centroid_y` | `observation.tactile.contact_geometry` | (3,) |
| `normal_mag`, `normal_var`, `shear_mag`, `shear_dir` | `observation.tactile.field` | (4,) |
| `delta_normal`, `delta_shear`, `friction_cone_ratio` | `observation.tactile.dynamics` | (3,) |
| tactile image (visual sensors only) | `observation.images.tactile` | (H, W, 3) video |

### Sensor-Specific Behavior

- **GelSight / DIGIT** (visual): All 22 dims + tactile images as video
- **PaXini** (force): 20 dims (no optical flow), no images
- **Daimon** (magnetic): 22 dims, no images
- **Touchd / UniVTAC / VTAC**: Depends on sensor config

## Custom Feature Config

Override the default feature mapping by providing a YAML config:

```bash
python convert_tlabel_to_lerobot.py \
    --input-dir ./data/ \
    --repo-id user/dataset \
    --config config/tlabel_default_features.yaml
```

See `config/tlabel_default_features.yaml` for the full default configuration.

## Integration with Training

Once converted, the dataset works with any LeRobot-compatible training pipeline:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("username/tactile_dataset")
print(dataset.features)  # Includes observation.tactile.* features
```

## References

- [TLabel GitHub](https://github.com/liesliy/tlabel)
- [TLabel PyPI](https://pypi.org/project/tlabel/)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [LeFlexiTac](https://tna001-ai.github.io/LeFlexiTac/index.html) — Tactile + LeRobot integration example
