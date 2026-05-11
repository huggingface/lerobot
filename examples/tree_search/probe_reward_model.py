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

from policy_inference_api import _load_task_language_map, _translate_task_language
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
    variant: str
    sample_index: int
    episode_index: int
    dataset_index: int
    local_index: int
    temporal_local_indices: list[int]
    frame_index: int | None
    timestamp: float | None
    task: str
    label: float
    prediction: float
    annotated_image_path: str


@dataclass
class ProbeVariant:
    name: str
    description: str
    local_indices: list[int]
    task_language: str
    video_path: Path


def _reader_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        force_cache_sync=args.force_cache_sync,
        download_videos=args.download_videos,
        video_backend=args.video_backend,
    )


def _variant_video_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.video_path is None:
        base = args.output_dir / "reward_probe.mp4"
    else:
        base = args.video_path
    return {
        "success": base.with_name(f"{base.stem}_success{base.suffix}"),
        "wrong_instruction": base.with_name(f"{base.stem}_wrong_instruction{base.suffix}"),
        "reversed_actions": base.with_name(f"{base.stem}_reversed_actions{base.suffix}"),
    }


def _get_wrong_task_language(
    args: argparse.Namespace,
    correct_original_task_language: str,
    task_language_translations: dict[str, str],
) -> str:
    if args.wrong_task_language is not None:
        return args.wrong_task_language

    if args.wrong_task_order is not None:
        wrong_original = str(_get_libero_task(args.suite, args.wrong_task_order)["language"])
        return _translate_task_language(
            wrong_original,
            suite=args.suite,
            task_id=int(args.wrong_task_order),
            translations=task_language_translations,
        )

    try:
        from libero.libero import benchmark
    except ImportError as exc:
        raise SystemExit("LIBERO is required to select a default wrong task instruction.") from exc

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.suite not in benchmark_dict:
        raise SystemExit(f"Unknown suite '{args.suite}'. Available: {sorted(benchmark_dict)}")

    suite = benchmark_dict[args.suite]()
    n_tasks = int(getattr(suite, "n_tasks", len(getattr(suite, "tasks", []))))
    if n_tasks <= 1:
        raise SystemExit(
            "Cannot auto-select a wrong task instruction from a suite with fewer than two tasks. "
            "Pass --wrong_task_language explicitly."
        )

    for offset in range(1, n_tasks):
        task_order = (int(args.task_order) + offset) % n_tasks
        candidate = str(suite.get_task(task_order).language)
        if candidate != correct_original_task_language:
            return _translate_task_language(
                candidate,
                suite=args.suite,
                task_id=task_order,
                translations=task_language_translations,
            )
    raise SystemExit("Could not find a wrong task instruction different from the selected task.")


def _make_probe_variants(
    args: argparse.Namespace,
    local_indices: list[int],
    *,
    task_language: str,
    wrong_task_language: str,
) -> list[ProbeVariant]:
    video_paths = _variant_video_paths(args)
    ordered_indices = [int(ix) for ix in local_indices]
    return [
        ProbeVariant(
            name="success",
            description="Ordered successful demonstration frames.",
            local_indices=ordered_indices,
            task_language=task_language,
            video_path=video_paths["success"],
        ),
        ProbeVariant(
            name="wrong_instruction",
            description=(
                "Same successful demonstration frames, but scored with an instruction for a different task."
            ),
            local_indices=ordered_indices,
            task_language=wrong_task_language,
            video_path=video_paths["wrong_instruction"],
        ),
        ProbeVariant(
            name="reversed_actions",
            description=(
                "Same successful demonstration frames in reverse temporal order, "
                "scored with the correct instruction."
            ),
            local_indices=list(reversed(ordered_indices)),
            task_language=task_language,
            video_path=video_paths["reversed_actions"],
        ),
    ]


def _checkpoint_arg(checkpoint: dict[str, Any], key: str, default: Any = None) -> Any:
    args = checkpoint.get("args")
    if isinstance(args, dict):
        return args.get(key, default)
    return default


def _resolve_scene_temporal_stride(args: argparse.Namespace, checkpoint: dict[str, Any]) -> int:
    if args.scene_temporal_stride is not None:
        return max(1, int(args.scene_temporal_stride))
    return max(1, int(_checkpoint_arg(checkpoint, "scene_temporal_stride", 10)))


def _scene_temporal_indices(
    *,
    variant: ProbeVariant,
    sample_index: int,
    local_index: int,
    window: int,
    stride: int,
) -> list[int]:
    window = max(1, int(window))
    stride = max(1, int(stride))
    if window == 1:
        return [int(local_index)]

    if variant.name == "reversed_actions":
        # For the reversed probe, build the temporal buffer in the presented
        # reversed order. This tests whether the model reacts to sequence order
        # instead of silently seeing normal demonstration history.
        return [
            int(variant.local_indices[max(0, sample_index - offset)])
            for offset in reversed(range(window))
        ]

    return [
        max(0, int(local_index) - stride * offset)
        for offset in reversed(range(window))
    ]


def _score_variant(
    *,
    variant: ProbeVariant,
    reader: Any,
    args: argparse.Namespace,
    model_cfg: Any,
    processor: RewardBatchProcessor,
    model: torch.nn.Module,
    device: torch.device,
    episode_index: int,
    denom: int,
    scene_temporal_stride: int,
) -> tuple[list[RewardProbeRecord], list[np.ndarray]]:
    records: list[RewardProbeRecord] = []
    video_frames: list[np.ndarray] = []
    frames_dir = args.output_dir / "annotated_frames" / variant.name
    frames_dir.mkdir(parents=True, exist_ok=True)

    for sample_index, local_index in enumerate(variant.local_indices):
        item = reader[local_index]
        temporal_local_indices = _scene_temporal_indices(
            variant=variant,
            sample_index=sample_index,
            local_index=local_index,
            window=model_cfg.scene_temporal_window,
            stride=scene_temporal_stride,
        )
        scene_images = [
            _image_to_uint8_hwc(reader[temporal_local_index][args.scene_camera_key])
            for temporal_local_index in temporal_local_indices
        ]
        scene_image = _image_to_uint8_hwc(item[args.scene_camera_key])
        wrist_image = (
            _image_to_uint8_hwc(item[args.wrist_camera_key])
            if args.wrist_camera_key and args.wrist_camera_key in item
            else None
        )
        sample = {
            "task": variant.task_language,
            "episode_index": episode_index,
            "local_index": local_index,
            "temporal_local_indices": temporal_local_indices,
            "label": float(local_index) / float(denom),
            "scene_image": scene_image,
            "scene_images": scene_images,
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
            reason=(
                f"{variant.name} target={sample['label']:.3f} "
                f"temporal={temporal_local_indices} task={variant.task_language}"
            ),
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
                variant=variant.name,
                sample_index=sample_index,
                episode_index=episode_index,
                dataset_index=dataset_index,
                local_index=local_index,
                temporal_local_indices=temporal_local_indices,
                frame_index=int(frame_index) if frame_index is not None else None,
                timestamp=float(timestamp) if timestamp is not None else None,
                task=variant.task_language,
                label=float(sample["label"]),
                prediction=prediction,
                annotated_image_path=annotated_path.relative_to(args.output_dir).as_posix(),
            )
        )
    return records, video_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_repo_id", default="HuggingFaceVLA/libero")
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task_order", type=int, required=True)
    parser.add_argument(
        "--task_language_map",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping from LIBERO suite/task ids or English task strings to replacement "
            "task text used by the reward model. Episode lookup still uses the original LIBERO task."
        ),
    )
    parser.add_argument("--episode_index", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument(
        "--scene_temporal_stride",
        type=int,
        default=None,
        help=(
            "Frame stride between scene images in the temporal buffer. "
            "Defaults to checkpoint training args when available, otherwise 10."
        ),
    )
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
    parser.add_argument(
        "--wrong_task_order",
        type=int,
        default=None,
        help="LIBERO task order to use as the wrong instruction. Defaults to the next task in the suite.",
    )
    parser.add_argument(
        "--wrong_task_language",
        default=None,
        help="Explicit wrong task instruction text. Overrides --wrong_task_order.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video_path in _variant_video_paths(args).values():
        video_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, model_cfg, checkpoint = load_reward_model_checkpoint(args.checkpoint, device=device)
    processor = RewardBatchProcessor(model_cfg)
    scene_temporal_stride = _resolve_scene_temporal_stride(args, checkpoint)
    task_language_translations = _load_task_language_map(args.task_language_map)
    logger.info(
        "Reward probe temporal config: scene_temporal_window=%s scene_temporal_stride=%s",
        model_cfg.scene_temporal_window,
        scene_temporal_stride,
    )

    task_metadata = _get_libero_task(args.suite, args.task_order)
    original_task_language = str(task_metadata["language"])
    task_language = _translate_task_language(
        original_task_language,
        suite=args.suite,
        task_id=int(args.task_order),
        translations=task_language_translations,
    )
    if task_language != original_task_language:
        logger.info("Using translated task text: original=%r translated=%r", original_task_language, task_language)
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
    wrong_task_language = _get_wrong_task_language(
        args,
        original_task_language,
        task_language_translations,
    )

    variants = _make_probe_variants(
        args,
        local_indices,
        task_language=task_language,
        wrong_task_language=wrong_task_language,
    )
    all_records: list[RewardProbeRecord] = []
    variant_summaries: dict[str, Any] = {}
    for variant in variants:
        logger.info("Scoring probe variant=%s description=%s", variant.name, variant.description)
        records, video_frames = _score_variant(
            variant=variant,
            reader=reader,
            args=args,
            model_cfg=model_cfg,
            processor=processor,
            model=model,
            device=device,
            episode_index=episode_index,
            denom=denom,
            scene_temporal_stride=scene_temporal_stride,
        )
        all_records.extend(records)
        scores = [record.prediction for record in records]
        variant_summaries[variant.name] = {
            "description": variant.description,
            "task": variant.task_language,
            "video_path": str(variant.video_path),
            "local_indices": variant.local_indices,
            "scene_temporal_window": model_cfg.scene_temporal_window,
            "scene_temporal_stride": scene_temporal_stride,
            "first_score": float(scores[0]) if scores else None,
            "last_score": float(scores[-1]) if scores else None,
            **_score_monotonicity(scores, tolerance=0.02),
        }
        if video_frames:
            write_video(str(variant.video_path), np.stack(video_frames), fps=args.video_fps)
            logger.info("Wrote probe video variant=%s path=%s", variant.name, variant.video_path)

    summary = {
        "checkpoint": str(args.checkpoint),
        "model_config": model_cfg.to_dict(),
        "checkpoint_training_args": checkpoint.get("args"),
        "dataset_repo_id": args.dataset_repo_id,
        "suite": args.suite,
        "task_order": args.task_order,
        "task": task_language,
        "original_task": original_task_language,
        "task_language_map": str(args.task_language_map) if args.task_language_map is not None else None,
        "task_language_map_keys": sorted(task_language_translations.keys()),
        "wrong_task_language": wrong_task_language,
        "episode_index": episode_index,
        "scene_temporal_window": model_cfg.scene_temporal_window,
        "scene_temporal_stride": scene_temporal_stride,
        "variants": variant_summaries,
    }
    payload = {
        "summary": _to_jsonable(summary),
        "frames": [_to_jsonable(asdict(record)) for record in all_records],
    }
    (args.output_dir / "reward_scores.json").write_text(json.dumps(payload, indent=2))
    with (args.output_dir / "reward_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "sample_index",
                "episode_index",
                "dataset_index",
                "local_index",
                "temporal_local_indices",
                "frame_index",
                "timestamp",
                "task",
                "label",
                "prediction",
                "annotated_image_path",
            ],
        )
        writer.writeheader()
        for record in all_records:
            row = asdict(record)
            writer.writerow(row)
    logger.info("Wrote reward scores: %s", args.output_dir / "reward_scores.json")


if __name__ == "__main__":
    main()
