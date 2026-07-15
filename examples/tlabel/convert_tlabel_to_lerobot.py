#!/usr/bin/env python3
"""
Convert TLabel tactile sensor datasets to LeRobotDataset v3.0 format.

Usage:
    python convert_tlabel_to_lerobot.py \
        --input-dir /path/to/tlabel_dataset/ \
        --repo-id username/tactile_dataset \
        --fps 30 \
        --sensor-type gelsight

Requirements:
    pip install tlabel lerobot
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("NumPy is required: pip install numpy")
    sys.exit(1)


# Default feature configuration for tactile data (22-dim unified schema)
DEFAULT_TACTILE_FEATURES = {
    "observation.tactile.contact": {
        "dtype": "float32",
        "shape": [1],
        "names": ["contact"],
    },
    "observation.tactile.force": {
        "dtype": "float32",
        "shape": [3],
        "names": ["magnitude", "direction", "peak"],
    },
    "observation.tactile.deformation": {
        "dtype": "float32",
        "shape": [2],
        "names": ["magnitude", "rate"],
    },
    "observation.tactile.slip": {
        "dtype": "float32",
        "shape": [2],
        "names": ["entropy", "event"],
    },
    "observation.tactile.texture": {
        "dtype": "float32",
        "shape": [1],
        "names": ["energy"],
    },
    "observation.tactile.contact_geometry": {
        "dtype": "float32",
        "shape": [3],
        "names": ["area", "centroid_x", "centroid_y"],
    },
    "observation.tactile.field": {
        "dtype": "float32",
        "shape": [4],
        "names": ["normal_mag", "normal_var", "shear_mag", "shear_dir"],
    },
    "observation.tactile.dynamics": {
        "dtype": "float32",
        "shape": [3],
        "names": ["delta_normal", "delta_shear", "friction_cone"],
    },
}

# Base LeRobot features (required for all datasets)
BASE_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}

# Sensor-specific feature overrides
SENSOR_CONFIGS = {
    "gelsight": {"has_image": True, "image_key": "observation.images.tactile"},
    "digit": {"has_image": True, "image_key": "observation.images.tactile"},
    "paxini": {"has_image": False},
    "daimon": {"has_image": False},
    "touchd": {"has_image": False},
    "univtac": {"has_image": False},
    "vtac": {"has_image": False},
}


def load_tlabel_data(input_dir: Path) -> dict:
    """Load TLabel exported dataset.

    Supports both JSON and CSV formats from tlabel.export().
    Returns a dict with episodes and their frame data.
    """
    try:
        import tlabel
        print(f"Loading TLabel data from {input_dir}...")
        dataset = tlabel.load(str(input_dir))
        return dataset
    except ImportError:
        print("tlabel not found. Install with: pip install tlabel")
        print("Falling back to manual JSON/CSV loading...")
        return _load_manual(input_dir)


def _load_manual(input_dir: Path) -> dict:
    """Manual loading fallback when tlabel package is not available."""
    data_file = input_dir / "tlabel_export.json"
    if not data_file.exists():
        csv_files = list(input_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No TLabel data found in {input_dir}. "
                "Expected tlabel_export.json or *.csv files."
            )
        return _load_csv_episodes(csv_files)

    with open(data_file) as f:
        raw = json.load(f)

    episodes = {}
    for frame in raw.get("frames", []):
        ep_idx = frame.get("episode_index", 0)
        if ep_idx not in episodes:
            episodes[ep_idx] = []
        episodes[ep_idx].append(frame)

    return {"episodes": episodes, "metadata": raw.get("metadata", {})}


def _load_csv_episodes(csv_files: list) -> dict:
    """Load episodes from CSV files."""
    import csv

    episodes = {}
    for csv_file in csv_files:
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep_idx = int(row.get("episode_index", 0))
                if ep_idx not in episodes:
                    episodes[ep_idx] = []
                frame = {}
                for k, v in row.items():
                    try:
                        frame[k] = float(v)
                    except (ValueError, TypeError):
                        frame[k] = v
                episodes[ep_idx].append(frame)

    return {"episodes": episodes, "metadata": {}}


def build_features(sensor_type: str, config_path: str = None, has_image: bool = False) -> dict:
    """Build the complete feature dict for LeRobotDataset."""
    features = {}
    features.update(BASE_FEATURES)

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            custom_features = yaml.safe_load(f)
        if custom_features:
            features.update(custom_features)
        else:
            features.update(DEFAULT_TACTILE_FEATURES)
    else:
        features.update(DEFAULT_TACTILE_FEATURES)

    sensor_config = SENSOR_CONFIGS.get(sensor_type, SENSOR_CONFIGS["gelsight"])
    if sensor_config.get("has_image") or has_image:
        image_key = sensor_config.get("image_key", "observation.images.tactile")
        features[image_key] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        }

    return features


def extract_tactile_features(frame_data: dict, sensor_type: str) -> dict:
    """Extract tactile feature values from a TLabel frame.

    Maps TLabel 22-dim schema to LeRobot feature keys.
    """
    features = {}

    features["observation.tactile.contact"] = [
        float(frame_data.get("contact", 0.0))
    ]

    features["observation.tactile.force"] = [
        float(frame_data.get("force_magnitude", 0.0)),
        float(frame_data.get("force_direction", 0.0)),
        float(frame_data.get("force_peak", 0.0)),
    ]

    features["observation.tactile.deformation"] = [
        float(frame_data.get("deformation_magnitude", 0.0)),
        float(frame_data.get("temporal_deformation_rate", 0.0)),
    ]

    features["observation.tactile.slip"] = [
        float(frame_data.get("slip_entropy", 0.0)),
        float(frame_data.get("slip_event", 0.0)),
    ]

    features["observation.tactile.texture"] = [
        float(frame_data.get("texture_energy", 0.0))
    ]

    features["observation.tactile.contact_geometry"] = [
        float(frame_data.get("contact_area", 0.0)),
        float(frame_data.get("centroid_x", 0.0)),
        float(frame_data.get("centroid_y", 0.0)),
    ]

    features["observation.tactile.field"] = [
        float(frame_data.get("normal_mag", 0.0)),
        float(frame_data.get("normal_var", 0.0)),
        float(frame_data.get("shear_mag", 0.0)),
        float(frame_data.get("shear_dir", 0.0)),
    ]

    features["observation.tactile.dynamics"] = [
        float(frame_data.get("delta_normal", 0.0)),
        float(frame_data.get("delta_shear", 0.0)),
        float(frame_data.get("friction_cone_ratio", 0.0)),
    ]

    return features


def convert(
    input_dir: Path,
    repo_id: str,
    fps: int = 30,
    sensor_type: str = "gelsight",
    output_dir: str = None,
    push_to_hub: bool = False,
    task: str = "tactile manipulation",
    config_path: str = None,
):
    """Main conversion function."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("lerobot not found. Install with: pip install lerobot")
        sys.exit(1)

    data = load_tlabel_data(input_dir)
    episodes = data["episodes"]
    print(f"Found {len(episodes)} episodes")

    has_image = False
    sensor_config = SENSOR_CONFIGS.get(sensor_type, {})
    if sensor_config.get("has_image", False):
        for ep_frames in episodes.values():
            if ep_frames and "tactile_image" in ep_frames[0]:
                has_image = True
                break

    features = build_features(sensor_type, config_path, has_image)
    use_videos = has_image

    print(f"Sensor: {sensor_type}")
    print(f"Features: {len(features)} keys")
    print(f"Video features: {use_videos}")

    root = output_dir or "./lerobot_data"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type=sensor_type,
        root=root,
        use_videos=use_videos,
    )

    for ep_idx, frames in sorted(episodes.items()):
        print(f"  Episode {ep_idx}: {len(frames)} frames")

        for frame_data in frames:
            frame = {}
            tactile = extract_tactile_features(frame_data, sensor_type)
            frame.update(tactile)
            frame["task"] = task

            if has_image and "tactile_image" in frame_data:
                img = frame_data["tactile_image"]
                if isinstance(img, str):
                    img_path = input_dir / img
                    if img_path.exists():
                        from PIL import Image
                        frame["observation.images.tactile"] = np.array(
                            Image.open(img_path)
                        )
                elif isinstance(img, np.ndarray):
                    frame["observation.images.tactile"] = img

            dataset.add_frame(frame)

        dataset.save_episode(task=task)

    dataset.finalize()
    print(f"\nDataset saved to: {root}/{repo_id}")

    if push_to_hub:
        print("Pushing to Hugging Face Hub...")
        dataset.push_to_hub()
        print(f"Pushed: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TLabel tactile data to LeRobotDataset format"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Path to TLabel exported dataset directory"
    )
    parser.add_argument(
        "--repo-id", type=str, required=True,
        help="LeRobot dataset repo ID (e.g., username/dataset_name)"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Sampling rate in Hz (default: 30)"
    )
    parser.add_argument(
        "--sensor-type", type=str, default="gelsight",
        choices=list(SENSOR_CONFIGS.keys()),
        help="Tactile sensor type (default: gelsight)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Local output directory (default: ./lerobot_data/)"
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--task", type=str, default="tactile manipulation",
        help="Task description string"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom feature config YAML"
    )

    args = parser.parse_args()
    convert(
        input_dir=Path(args.input_dir),
        repo_id=args.repo_id,
        fps=args.fps,
        sensor_type=args.sensor_type,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        task=args.task,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
