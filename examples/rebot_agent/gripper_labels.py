# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Prepare a real-image gripper review pack and export reviewed per-arm DETR labels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.annotations.steerable_pipeline.frames import VideoFrameProvider, _frame_to_pil
from lerobot.annotations.steerable_pipeline.reader import iter_episodes
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

ARMS = ("left", "right")
CAMERAS = tuple(f"observation.images.{name}" for name in ("base", "left_wrist", "right_wrist"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_context(pack_path: Path, context_path: Path) -> dict:
    """Validate companion views without changing the pack or human-label identity."""
    pack = json.loads(pack_path.read_text())
    context = json.loads(context_path.read_text())
    if context.get("version") != 1 or context.get("pack_sha256") != digest(pack_path):
        raise ValueError("Context does not match this image pack")
    rows = {row["id"]: row for row in context["images"]}
    if len(rows) != len(context["images"]) or set(rows) != {im["id"] for im in pack["images"]}:
        raise ValueError("Context must contain every pack image exactly once")
    for image in pack["images"]:
        row = rows[image["id"]]
        if any(row[k] != image[k] for k in ("episode_index", "frame_index", "timestamp")):
            raise ValueError("Context must come from the same recorded frame")
        if row.get("image_sha256") != image["sha256"]:
            raise ValueError("Context target image changed")
        if digest(pack_path.parent / image["file_name"]) != image["sha256"]:
            raise ValueError("A source image changed after pack creation")
        views = row["views"]
        if len(views) != 2 or {v["camera"] for v in views} != set(CAMERAS) - {image["camera"]}:
            raise ValueError("Context requires the other two camera views")
        for view in views:
            name = Path(view["file_name"])
            path = (pack_path.parent / name).resolve()
            if name.is_absolute() or not path.is_relative_to(pack_path.parent.resolve()):
                raise ValueError("Context images must be stored inside the review pack")
            if digest(path) != view["sha256"]:
                raise ValueError("Context image bytes changed")
            with Image.open(path) as frame:
                if frame.size != (view["width"], view["height"]):
                    raise ValueError("Context image dimensions changed")
    return {**context, "source_sha256": digest(context_path)}


def prepare_context(root: Path, pack_path: Path, output: Path) -> dict:
    """Add synchronized camera context while preserving the original annotation pack."""
    pack_path, output = pack_path.resolve(), output.resolve()
    if output.parent != pack_path.parent:
        raise ValueError("Write context beside the image pack")
    pack = json.loads(pack_path.read_text())
    source = json.loads((root / "source.json").read_text())
    if any(source[k] != pack["source"][k] for k in ("repo_id", "revision")):
        raise ValueError("Context requires the same pinned source dataset")
    episodes = tuple(sorted({im["episode_index"] for im in pack["images"]}))
    found = list(iter_episodes(root, only_episodes=episodes))
    records = {r.episode_index: r for r in found}
    if len(found) != len(episodes) or set(records) != set(episodes):
        raise ValueError("Context source episodes are missing or duplicated")
    for image in pack["images"]:
        record = records[image["episode_index"]]
        indices = np.flatnonzero(np.asarray(record.frame_indices) == image["frame_index"])
        if len(indices) != 1 or float(record.frame_timestamps[indices[0]]) != image["timestamp"]:
            raise ValueError("Context source frame/timestamp does not match the pack")
        if image["camera"] not in CAMERAS or digest(pack_path.parent / image["file_name"]) != image["sha256"]:
            raise ValueError("Invalid or changed annotation image")
    directory = output.with_suffix("")
    if output.exists() or directory.exists():
        raise ValueError("Use a fresh context output path")
    provider = VideoFrameProvider(root, video_backend="pyav", cache_size=8)
    directory.mkdir()
    context = {"version": 1, "pack_sha256": digest(pack_path), "images": []}
    for image in pack["images"]:
        row = {k: image[k] for k in ("id", "episode_index", "frame_index", "timestamp")}
        row.update(image_sha256=image["sha256"], views=[])
        for camera in CAMERAS:
            if camera == image["camera"]:
                continue
            frames = provider.frames_at(records[image["episode_index"]], [image["timestamp"]], camera)
            if len(frames) != 1:
                raise ValueError("Missing synchronized context frame")
            frame = _frame_to_pil(frames[0]).convert("RGB")
            path = directory / f"{image['id']:04d}_{camera.rsplit('.', 1)[-1]}.jpg"
            frame.save(path, quality=95)
            row["views"].append(
                {
                    "camera": camera,
                    "file_name": path.relative_to(pack_path.parent).as_posix(),
                    "sha256": digest(path),
                    "width": frame.width,
                    "height": frame.height,
                }
            )
        context["images"].append(row)
    output.write_text(json.dumps(context, indent=2) + "\n")
    read_context(pack_path, output)
    return context


def render_review(
    pack_path: Path,
    output: Path,
    suggestions_path: Path | None = None,
    context_path: Path | None = None,
):
    """Show unassigned model boxes without modifying any human labels or confirmations."""
    if output.resolve().suffix != ".html" or output.resolve().parent != pack_path.resolve().parent:
        raise ValueError("Write the review HTML beside its image pack")
    pack = json.loads(pack_path.read_text())
    suggestions = None
    if suggestions_path is not None:
        suggestions = json.loads(suggestions_path.read_text())
        if (
            suggestions.get("kind") != "model_proposals"
            or suggestions.get("pack_sha256") != digest(pack_path)
            or not isinstance(suggestions.get("reviewer"), str)
            or not suggestions["reviewer"].strip()
        ):
            raise ValueError("Suggestions require model attribution and the matching image pack")
        by_id = {image["id"]: image for image in pack["images"]}
        seen = set()
        for row in suggestions["images"]:
            if row["id"] not in by_id or row["id"] in seen:
                raise ValueError("Unknown or duplicate suggestion image")
            seen.add(row["id"])
            image = by_id[row["id"]]
            if (
                row["image_sha256"] != image["sha256"]
                or digest(pack_path.parent / image["file_name"]) != image["sha256"]
            ):
                raise ValueError("Suggestion image differs from the reviewed source")
            for candidate in row["candidates"]:
                box = candidate["bbox_xyxy"]
                if (
                    not isinstance(box, list)
                    or len(box) != 4
                    or any(type(v) not in (int, float) for v in box)
                    or not np.isfinite(box).all()
                    or not (
                        0 <= box[0] < box[2] <= image["width"] and 0 <= box[1] < box[3] <= image["height"]
                    )
                    or "arm" in candidate
                ):
                    raise ValueError(
                        "Suggestions must be valid unassigned boxes in original-image coordinates"
                    )
        suggestions = {**suggestions, "source_sha256": digest(suggestions_path)}
    payload = json.dumps({"pack_sha256": digest(pack_path), "pack": pack}).replace("<", "\\u003c")
    suggestion_payload = json.dumps(suggestions).replace("<", "\\u003c")
    context = read_context(pack_path, context_path) if context_path is not None else None
    context_payload = json.dumps(context).replace("<", "\\u003c")
    template = Path(__file__).with_name("gripper_review.html").read_text()
    output.write_text(
        template.replace("/* PACK_DATA */ null", payload)
        .replace("/* MODEL_SUGGESTIONS */ null", suggestion_payload)
        .replace("/* CAMERA_CONTEXT */ null", context_payload)
    )


def prepare(root: Path, output: Path, episodes: list[int], per_episode: int = 4):
    """Stratify views/times and split detector evaluation by episode, outside VLA holdout."""
    source = json.loads((root / "source.json").read_text())
    if not source.get("repo_id") or not source.get("revision"):
        raise ValueError("A pinned dataset source.json is required")
    info = json.loads((root / "meta/info.json").read_text())
    holdout = list(range(info["total_episodes"] - 10, info["total_episodes"]))
    if len(episodes) < 5 or len(set(episodes)) != len(episodes) or per_episode < 1:
        raise ValueError("Select at least five distinct episodes and a positive frame count")
    if set(episodes) & set(holdout):
        raise ValueError("VLA-held-out episodes must not fit the gripper detector")
    records = list(iter_episodes(root, only_episodes=tuple(episodes)))
    if sorted(record.episode_index for record in records) != sorted(episodes):
        raise ValueError("Requested episodes are missing or duplicated across shards")
    provider = VideoFrameProvider(root, video_backend="pyav", cache_size=8)
    cameras = [f"observation.images.{name}" for name in ("base", "left_wrist", "right_wrist")]
    if not set(cameras).issubset(provider.camera_keys):
        raise ValueError("The ReBot label pack requires base and both wrist cameras")
    metadata = LeRobotDatasetMetadata(repo_id=source["repo_id"], root=root)
    missing = sorted(
        {
            str(metadata.get_video_file_path(ep, camera))
            for ep in episodes
            for camera in cameras
            if not (root / metadata.get_video_file_path(ep, camera)).exists()
        }
    )
    if missing:
        raise ValueError(f"Download the required video shards before preparing labels: {missing}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    manifest = {"version": 1, "source": source, "vla_holdout_episodes": holdout, "images": []}
    for ep_offset, record in enumerate(sorted(records, key=lambda r: r.episode_index)):
        indices = np.linspace(
            0.1 * (record.row_count - 1), 0.9 * (record.row_count - 1), per_episode, dtype=int
        )
        for offset, row_index in enumerate(indices):
            camera = cameras[(ep_offset + offset) % len(cameras)]
            timestamp = float(record.frame_timestamps[row_index])
            frames = provider.frames_at(record, [timestamp], camera)
            if len(frames) != 1:
                raise ValueError(f"Missing source image for episode {record.episode_index}: {camera}")
            image = _frame_to_pil(frames[0]).convert("RGB")
            image_id = len(manifest["images"])
            filename = f"images/{image_id:04d}.jpg"
            image.save(output / filename, quality=95)
            manifest["images"].append(
                {
                    "id": image_id,
                    "file_name": filename,
                    "sha256": digest(output / filename),
                    "width": image.width,
                    "height": image.height,
                    "episode_index": record.episode_index,
                    "frame_index": int(record.frame_indices[row_index]),
                    "timestamp": timestamp,
                    "camera": camera,
                    "split": "validation" if ep_offset % 5 == 4 else "train",
                }
            )
    pack_path = output / "pack.json"
    pack_path.write_text(json.dumps(manifest, indent=2) + "\n")
    render_review(pack_path, output / "review.html")
    return manifest


def export_coco(pack_path: Path, labels_path: Path, output: Path) -> dict:
    """Never turn unreviewed/uncertain arms into empty negative detector targets."""
    manifest = json.loads(pack_path.read_text())
    labels = json.loads(labels_path.read_text())
    if labels.get("version") != 1 or labels.get("pack_sha256") != digest(pack_path):
        raise ValueError("Labels do not match this exact image pack")
    by_id = {row["id"]: row for row in labels["images"]}
    if len(by_id) != len(labels["images"]) or set(by_id) != {im["id"] for im in manifest["images"]}:
        raise ValueError("Labels must contain every pack image exactly once")
    categories = [{"id": i, "name": f"{arm}_gripper"} for i, arm in enumerate(ARMS)]
    datasets = {
        split: {"images": [], "annotations": [], "categories": categories}
        for split in ("train", "validation")
    }
    excluded = []
    split_episodes = {split: set() for split in datasets}
    for image in manifest["images"]:
        if image["episode_index"] in manifest["vla_holdout_episodes"]:
            raise ValueError("Pack contains a VLA-held-out episode")
        if digest(pack_path.parent / image["file_name"]) != image["sha256"]:
            raise ValueError("A source image changed after pack creation")
        row = by_id[image["id"]]
        review = row.get("review", {})
        if review.get("kind") != "human" or not review.get("reviewer") or not review.get("confirmed"):
            raise ValueError(f"Image {image['id']} requires attributed human review")
        if set(row.get("arms", {})) != set(ARMS):
            raise ValueError("Both physical arms must have explicit visibility labels")
        annotations = []
        uncertain = False
        for category, arm in enumerate(ARMS):
            label = row["arms"][arm]
            visibility = label.get("visibility")
            if visibility not in ("visible", "not_visible", "uncertain"):
                raise ValueError("Unlabeled arms cannot become negative examples")
            uncertain |= visibility == "uncertain"
            box = label.get("bbox_xyxy")
            if visibility != "visible":
                if box is not None:
                    raise ValueError("Non-visible/uncertain grippers must not have boxes")
                continue
            if not review.get("arm_identity_verified"):
                raise ValueError("Verify physical arm identity, not its image-left/image-right location")
            if not isinstance(box, list) or len(box) != 4 or not all(type(v) in (int, float) for v in box):
                raise ValueError("Visible grippers require numeric XYXY boxes")
            x1, y1, x2, y2 = box
            if not np.isfinite(box).all() or not (
                0 <= x1 < x2 <= image["width"] and 0 <= y1 < y2 <= image["height"]
            ):
                raise ValueError("Gripper box is invalid or outside the original image")
            annotations.append(
                {
                    "image_id": image["id"],
                    "category_id": category,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
        if uncertain:
            excluded.append(image["id"])
            continue
        split = image["split"]
        split_episodes[split].add(image["episode_index"])
        dataset = datasets[split]
        dataset["images"].append(image)
        for annotation in annotations:
            dataset["annotations"].append({"id": len(dataset["annotations"]), **annotation})
    if split_episodes["train"] & split_episodes["validation"]:
        raise ValueError("Detector train/validation episodes overlap")
    for split, dataset in datasets.items():
        if {a["category_id"] for a in dataset["annotations"]} != {0, 1}:
            raise ValueError(f"{split} needs visible reviewed examples for each physical arm")
    report = {
        "pack_sha256": digest(pack_path),
        "labels_sha256": digest(labels_path),
        "excluded_uncertain_images": excluded,
        "splits": {
            k: {
                "images": len(v["images"]),
                "boxes": len(v["annotations"]),
                "episodes": sorted(split_episodes[k]),
            }
            for k, v in datasets.items()
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_pack.json").write_bytes(pack_path.read_bytes())
    (output / "reviewed_labels.json").write_bytes(labels_path.read_bytes())
    for split, dataset in datasets.items():
        dataset["info"] = {
            "source": manifest["source"],
            "image_root": str(pack_path.parent.resolve()),
            **report,
        }
        (output / f"{split}.json").write_text(json.dumps(dataset, indent=2) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("prepare")
    pack.add_argument("--dataset-root", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--episodes", nargs="+", type=int, default=list(range(25)))
    pack.add_argument("--per-episode", type=int, default=4)
    export = commands.add_parser("export")
    export.add_argument("--pack", type=Path, required=True)
    export.add_argument("--labels", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("review")
    review.add_argument("--pack", type=Path, required=True)
    review.add_argument("--suggestions", type=Path)
    review.add_argument("--context", type=Path)
    review.add_argument(
        "--output", type=Path, required=True, help="HTML beside the pack, so image paths resolve"
    )
    context = commands.add_parser("context", help="Add the other camera views at each annotation frame")
    context.add_argument("--dataset-root", type=Path, required=True)
    context.add_argument("--pack", type=Path, required=True)
    context.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.dataset_root, args.output, args.episodes, args.per_episode)
        print(f"Prepared {len(result['images'])} images. Open {args.output / 'review.html'}.")
    elif args.command == "export":
        print(json.dumps(export_coco(args.pack, args.labels, args.output), indent=2))
    elif args.command == "context":
        result = prepare_context(args.dataset_root, args.pack, args.output)
        print(f"Prepared synchronized camera context for {len(result['images'])} annotation images.")
    else:
        render_review(args.pack, args.output, args.suggestions, args.context)


if __name__ == "__main__":
    main()
