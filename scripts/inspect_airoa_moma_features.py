#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.constants import ACTION, OBS_STATE


def _tensor_summary(value: Any) -> tuple[str, tuple[int, ...]]:
    if isinstance(value, torch.Tensor):
        return str(value.dtype), tuple(value.shape)
    if isinstance(value, np.ndarray):
        return str(value.dtype), tuple(value.shape)
    tensor = torch.as_tensor(value)
    return str(tensor.dtype), tuple(tensor.shape)


def _load_sample(
    dataset_repo_id: str,
    dataset_root: Path | None,
    sample_index: int,
) -> tuple[dict[str, Any], str, str | None]:
    dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=dataset_root,
        download_videos=False,
    )

    try:
        sample = dataset[sample_index]
        return sample, "dataset", None
    except Exception as exc:  # noqa: BLE001
        sample = dataset.hf_dataset[sample_index]
        return sample, "hf_dataset", str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect AIRoA MoMa feature keys and tensor shapes from LeRobot dataset metadata/sample."
    )
    parser.add_argument("--dataset_repo_id", type=str, default=None, help="HF dataset repo id (e.g. org/name).")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=None,
        help="Local dataset root path (preferred when dataset access is gated).",
    )
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index to inspect.")
    args = parser.parse_args()

    if args.dataset_repo_id is None and args.dataset_root is None:
        parser.error("--dataset_repo_id と --dataset_root の少なくとも一方を指定してください。")

    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root is not None else None
    dataset_repo_id = args.dataset_repo_id

    if dataset_root is not None and dataset_repo_id is None:
        dataset_repo_id = dataset_root.name

    if dataset_repo_id is None:
        raise ValueError("dataset_repo_id を決定できませんでした。")

    metadata = LeRobotDatasetMetadata(repo_id=dataset_repo_id, root=dataset_root)
    features = dataset_to_policy_features(metadata.features)

    observation_keys = sorted([key for key in features if key.startswith("observation.")])
    action_keys = sorted([key for key, feature in features.items() if feature.type == FeatureType.ACTION])
    image_keys = sorted(
        [
            key
            for key, feature in features.items()
            if key.startswith("observation.") and feature.type == FeatureType.VISUAL
        ]
    )
    primary_action_key: str | None = None
    if ACTION in features:
        primary_action_key = ACTION
    elif action_keys:
        primary_action_key = next((key for key in action_keys if not key.endswith(".is_fresh")), action_keys[0])

    print("=== Dataset Source ===")
    print(f"dataset_repo_id: {dataset_repo_id}")
    print(f"dataset_root: {dataset_root if dataset_root is not None else '(default cache)'}")

    print("\n=== Available Keys ===")
    print("observation.* keys:")
    for key in observation_keys:
        print(f"  - {key}")

    print("action keys:")
    for key in action_keys:
        print(f"  - {key}")

    print("\n=== Image Key Candidates ===")
    for key in image_keys:
        print(f"  - {key}")

    state_shape = features[OBS_STATE].shape if OBS_STATE in features else None
    action_shape = features[primary_action_key].shape if primary_action_key in features else None

    print("\n=== Key Shapes (from metadata features) ===")
    print(f"{OBS_STATE}: {state_shape}")
    print(f"{primary_action_key}: {action_shape}")

    sample, sample_source, fallback_error = _load_sample(
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        sample_index=args.sample_index,
    )

    print("\n=== Sample Tensor Summary ===")
    print(f"sample_source: {sample_source}")
    if fallback_error is not None:
        print(f"note: dataset[sample_index] の読み出しに失敗したため hf_dataset を使用: {fallback_error}")

    keys_to_report = [OBS_STATE]
    if primary_action_key is not None:
        keys_to_report.append(primary_action_key)
    keys_to_report.extend(image_keys)
    seen: set[str] = set()
    for key in keys_to_report:
        if key in seen:
            continue
        seen.add(key)

        if key not in sample:
            print(f"  - {key}: (missing in sample)")
            continue

        dtype, shape = _tensor_summary(sample[key])
        print(f"  - {key}: dtype={dtype}, shape={shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
