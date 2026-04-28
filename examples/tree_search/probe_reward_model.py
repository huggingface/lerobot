#!/usr/bin/env python
"""Probe a trained reward model on a LIBERO demonstration episode."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.datasets.image_writer import write_image
from lerobot.utils.io_utils import write_video

from reward_model import RewardBatchProcessor, load_reward_model_checkpoint, move_batch_to_device
from train_reward_model import find_episodes_for_task, model_forward
from vlm_sequence_prompt_probe import (
    _coerce_scalar,
    _get_libero_task,
    _image_to_uint8_hwc,
    _load_episode_reader,
    _make_composite_frame,
    _sample_local_indices,
    _score_monotonicity,
    _to_jsonable,
)

logger = logging.getLogger(__name__)


@dataclass
class RewardProbeRecord:
    sample_index: int
    episode_index: int
    dataset_index: int
    local_index: int
    frame_index: int | None
    timestamp: float | None
    task: str
    label: float
    prediction: float
    annotated_image_path: str


def _reader_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        force_cache_sync=args.force_cache_sync,
        download_videos=args.download_videos,
        video_backend=args.video_backend,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_repo_id", default="HuggingFaceVLA/libero")
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task_order", type=int, required=True)
    parser.add_argument("--episode_index", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--scene_camera_key", default="observation.images.image")
    parser.add_argument("--wrist_camera_key", default="observation.images.image2")
    parser.add_argument("--state_key", default="observation.state")
    parser.add_argument("--download_videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_cache_sync", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video_backend", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--video_path", type=Path, default=None)
    parser.add_argument("--video_fps", type=int, default=2)
    parser.add_argument("--tile_width", type=int, default=256)
    parser.add_argument("--tile_height", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "annotated_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.video_path is None:
        args.video_path = args.output_dir / "reward_probe.mp4"
    args.video_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, model_cfg, checkpoint = load_reward_model_checkpoint(args.checkpoint, device=device)
    processor = RewardBatchProcessor(model_cfg)

    task_metadata = _get_libero_task(args.suite, args.task_order)
    task_language = str(task_metadata["language"])
    meta = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
        revision=args.dataset_revision,
    )
    episode_index = args.episode_index
    if episode_index is None:
        episode_index = find_episodes_for_task(meta, task_metadata=task_metadata, limit=1)[0]

    reader = _load_episode_reader(_reader_args(args), meta, episode_index)
    local_indices = _sample_local_indices(
        len(reader),
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    denom = max(1, len(reader) - 1)

    records: list[RewardProbeRecord] = []
    video_frames: list[np.ndarray] = []
    for sample_index, local_index in enumerate(local_indices):
        item = reader[local_index]
        scene_image = _image_to_uint8_hwc(item[args.scene_camera_key])
        wrist_image = (
            _image_to_uint8_hwc(item[args.wrist_camera_key])
            if args.wrist_camera_key and args.wrist_camera_key in item
            else None
        )
        sample = {
            "task": task_language,
            "episode_index": episode_index,
            "local_index": local_index,
            "label": float(local_index) / float(denom),
            "scene_image": scene_image,
            "wrist_image": wrist_image,
        }
        if model_cfg.use_proprioception:
            if args.state_key not in item:
                raise KeyError(f"Missing proprioception key '{args.state_key}' in dataset item.")
            sample["proprioception"] = item[args.state_key]
        with torch.no_grad():
            batch = move_batch_to_device(processor([sample]), device)
            prediction = float(model_forward(model, batch).detach().cpu()[0])

        dataset_index = int(_coerce_scalar(item.get("index")) or (reader.dataset_from_index + local_index))
        frame_index = _coerce_scalar(item.get("frame_index"))
        timestamp = _coerce_scalar(item.get("timestamp"))
        camera_images = [("scene", scene_image)]
        if wrist_image is not None:
            camera_images.append(("wrist", wrist_image))
        annotated = _make_composite_frame(
            camera_images,
            score=prediction,
            reason=f"target={sample['label']:.3f}",
            local_index=local_index,
            dataset_index=dataset_index,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
        )
        annotated_path = frames_dir / f"frame_{sample_index:04d}_idx_{dataset_index:08d}.png"
        write_image(annotated, annotated_path)
        video_frames.append(annotated)
        records.append(
            RewardProbeRecord(
                sample_index=sample_index,
                episode_index=episode_index,
                dataset_index=dataset_index,
                local_index=local_index,
                frame_index=int(frame_index) if frame_index is not None else None,
                timestamp=float(timestamp) if timestamp is not None else None,
                task=task_language,
                label=float(sample["label"]),
                prediction=prediction,
                annotated_image_path=annotated_path.relative_to(args.output_dir).as_posix(),
            )
        )
        logger.info(
            "frame=%s local_index=%s label=%.3f prediction=%.3f",
            sample_index,
            local_index,
            float(sample["label"]),
            prediction,
        )

    scores = [record.prediction for record in records]
    summary = {
        "checkpoint": str(args.checkpoint),
        "model_config": model_cfg.to_dict(),
        "checkpoint_training_args": checkpoint.get("args"),
        "dataset_repo_id": args.dataset_repo_id,
        "suite": args.suite,
        "task_order": args.task_order,
        "task": task_language,
        "episode_index": episode_index,
        **_score_monotonicity(scores, tolerance=0.02),
    }
    payload = {
        "summary": _to_jsonable(summary),
        "frames": [_to_jsonable(asdict(record)) for record in records],
    }
    (args.output_dir / "reward_scores.json").write_text(json.dumps(payload, indent=2))
    with (args.output_dir / "reward_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "episode_index",
                "dataset_index",
                "local_index",
                "frame_index",
                "timestamp",
                "label",
                "prediction",
                "annotated_image_path",
            ],
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row.pop("task")
            writer.writerow(row)

    if video_frames:
        write_video(str(args.video_path), np.stack(video_frames), fps=args.video_fps)
        logger.info("Wrote probe video: %s", args.video_path)
    logger.info("Wrote reward scores: %s", args.output_dir / "reward_scores.json")


if __name__ == "__main__":
    main()
