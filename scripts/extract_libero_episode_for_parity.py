#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""Extract one LIBERO episode for Robometer parity testing.

Loads a LeRobot LIBERO (or any video-bearing LeRobot) dataset, picks one
episode, samples ``--num-frames`` frames uniformly across its duration
(matching upstream Robometer's default of 8 frames), and saves them to
``.npz`` plus a sidecar ``.txt`` task file.

The ``.npz`` layout (``frames`` key, ``(T, H, W, C) uint8``) is what upstream
``example_inference_local.py`` consumes, so the same file feeds both pipelines
and frame sampling cannot drift.

Workflow:

1. Run this script (LeRobot env) to produce ``frames.npz`` + ``task.txt``.
2. Pass them to upstream ``scripts/example_inference_local.py``
   (upstream env) to produce reference progress / success outputs.
3. Pass the same ``frames.npz`` to ``scripts/parity_robometer.py``
   (LeRobot env) to compare both sides.

Example:

    uv run python scripts/extract_libero_episode_for_parity.py \\
        --repo-id lerobot/libero_10_image \\
        --episode 0 \\
        --num-frames 8 \\
        --out-dir /tmp/libero_ep0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _pick_visual_feature(features: dict, requested: str | None) -> str:
    """Return a visual feature key, preferring ``requested`` when given."""
    visual_keys = [
        key
        for key, ft in features.items()
        if getattr(ft, "type", None) == FeatureType.VISUAL or ft.get("dtype", "") == "video"
    ]
    if not visual_keys:
        raise ValueError(f"Dataset has no visual feature; available: {list(features)}")
    if requested is not None:
        if requested not in visual_keys:
            raise ValueError(f"Camera key {requested!r} not in dataset visual features {visual_keys}")
        return requested
    return visual_keys[0]


def _frame_uint8_hwc(tensor: torch.Tensor) -> np.ndarray:
    """Convert a LeRobotDataset video frame to ``uint8`` ``(H, W, C)`` RGB."""
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 + 1e-3 else arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        default="lerobot/libero_10_image",
        help="LeRobot LIBERO (or other) dataset repo id (default: lerobot/libero_10_image).",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index.")
    parser.add_argument(
        "--camera-key",
        default=None,
        help="Visual feature key (e.g. observation.images.image). Auto-selects first if omitted.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames to sample uniformly (default: 8 — Robometer's training-time default).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/robometer_parity/libero"),
        help="Directory to write frames.npz / task.txt / frame_indices.npy.",
    )
    args = parser.parse_args()

    print(f"Loading {args.repo_id} (episode {args.episode})...")
    dataset = LeRobotDataset(args.repo_id, episodes=[args.episode])

    camera_key = _pick_visual_feature(dataset.features, args.camera_key)
    print(f"Using camera key: {camera_key}")

    ep_from = int(dataset.episode_data_index["from"][0].item())
    ep_to = int(dataset.episode_data_index["to"][0].item())
    total_frames = ep_to - ep_from
    if total_frames <= 0:
        print(f"ERROR: episode {args.episode} has no frames.", file=sys.stderr)
        return 1
    print(f"Episode has {total_frames} frames; sampling {args.num_frames} uniformly.")

    indices = np.linspace(0, total_frames - 1, num=min(args.num_frames, total_frames), dtype=int)
    frames: list[np.ndarray] = []
    task: str = ""
    for offset in indices:
        sample = dataset[ep_from + int(offset)]
        frame_tensor = sample[camera_key]
        frames.append(_frame_uint8_hwc(frame_tensor))
        if not task:
            task = sample.get("task", "") or ""

    if not task:
        print("ERROR: episode has no task description in metadata.", file=sys.stderr)
        return 1

    frames_array = np.stack(frames)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = args.out_dir / "frames.npz"
    task_path = args.out_dir / "task.txt"
    indices_path = args.out_dir / "frame_indices.npy"

    np.savez(frames_path, frames=frames_array)
    task_path.write_text(task + "\n", encoding="utf-8")
    np.save(indices_path, indices)

    print()
    print(f"Wrote {frames_path} (shape={frames_array.shape}, dtype={frames_array.dtype})")
    print(f"Wrote {task_path}   (task={task!r})")
    print(f"Wrote {indices_path} (frame_indices={indices.tolist()})")
    print()
    print("Next steps:")
    print("  # in upstream env (where `robometer` is importable):")
    print(
        f"  python third_party/robometer/scripts/example_inference_local.py \\\n"
        f"      --model-path robometer/Robometer-4B \\\n"
        f"      --video {frames_path} \\\n"
        f'      --task "{task}" \\\n'
        f"      --out {args.out_dir / 'upstream.npy'}"
    )
    print()
    print("  # back in LeRobot env:")
    print(
        f"  uv run python scripts/parity_robometer.py \\\n"
        f"      --frames {frames_path} \\\n"
        f'      --task "{task}" \\\n'
        f"      --upstream-progress {args.out_dir / 'upstream.npy'} \\\n"
        f"      --upstream-success  {args.out_dir / 'upstream_success_probs.npy'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
