# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Export visual candidates into a derived dataset's native LeRobot language columns.

VQA events contain boxes and points; trace events contain observed object trajectories.
These are unreviewed extractor evidence, not accepted steering commands or gripper paths.
The source dataset is never rewritten. Videos are referenced locally through a symlink.
"""

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
from lerobot.datasets.language import language_events_arrow_type, language_feature_info, validate_camera_field
from lerobot.utils.constants import LANGUAGE_EVENTS


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def coordinates(value, size, *, box=False):
    if value is None:
        return None
    limits = size * (2 if box else 1)
    if len(value) != len(limits) or any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or v < 0
        or (v > limit if box else v >= limit)
        for v, limit in zip(value, limits, strict=True)
    ):
        raise ValueError(f"Invalid {'box' if box else 'point'} coordinates: {value}")
    if box and (value[0] >= value[2] or value[1] >= value[3]):
        raise ValueError("Box must have positive area")
    return value


def collect_candidates(extractions: list[Path], source: dict):
    """Aggregate by source frame/camera so native scalar VQA resolvers stay unambiguous."""
    events = {}
    provenance = []
    seen = set()
    for root in extractions:
        manifest = read_json(root / "extraction.json")
        manifest_hash = digest(root / "extraction.json")
        if manifest_hash in seen:
            raise ValueError("Duplicate extraction")
        seen.add(manifest_hash)
        if manifest["source"] != source:
            raise ValueError("Extraction and dataset source revisions differ")
        provenance.append(
            {
                "manifest_sha256": manifest_hash,
                "manifest": manifest,
                "clips": [],
                "artifact_root": str(root.resolve()),
            }
        )
        for clip in manifest["clips"]:
            directory = root / clip["path"]
            selected = read_json(directory / "task_objects.json")
            if selected["source_manifest_sha256"] != manifest_hash or selected[
                "source_tracks_sha256"
            ] != digest(directory / "tracks.json"):
                raise ValueError("Stale task-object filter")
            names = {obj["object_id"]: obj for obj in selected["objects"]}
            if len(names) != len(selected["objects"]):
                raise ValueError("Duplicate object IDs")
            frames = selected["frames"]
            if [{k: f[k] for k in ("frame_index", "timestamp", "sha256")} for f in frames] != clip["frames"]:
                raise ValueError("Tracked frames do not match extraction manifest")
            size = clip["image_size"]
            if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
                raise ValueError("Invalid image size")
            validate_camera_field("vqa", clip["camera"])
            if not frames or any(
                a["frame_index"] >= b["frame_index"] for a, b in zip(frames, frames[1:], strict=False)
            ):
                raise ValueError("Frames must be strictly ordered")
            if not names:
                continue
            history = defaultdict(list)
            provenance[-1]["clips"].append(
                {"path": clip["path"], "filter_sha256": digest(directory / "task_objects.json")}
            )
            for index, frame in enumerate(frames):
                if not clip["start_frame"] <= frame["frame_index"] < clip["end_frame"]:
                    raise ValueError("Frame outside clip interval")
                image = directory / "frames" / f"{index:06d}.jpg"
                if digest(image) != frame["sha256"]:
                    raise ValueError("Source image hash mismatch")
                key = (clip["episode_index"], frame["frame_index"], clip["camera"])
                bucket = events.setdefault(
                    key, {"timestamp": frame["timestamp"], "image_size": size, "detections": [], "traces": []}
                )
                if bucket["timestamp"] != frame["timestamp"] or bucket["image_size"] != size:
                    raise ValueError("Conflicting frame geometry or time")
                objects = {obj["object_id"]: obj for obj in frame["objects"]}
                if set(objects) != set(names) or len(objects) != len(frame["objects"]):
                    raise ValueError(
                        "Missing or duplicate object records; represent missing masks explicitly"
                    )
                for object_id, candidate in names.items():
                    obj = objects[object_id]
                    point = coordinates(obj["centroid"], size)
                    box = coordinates(obj["bbox_xyxy"], size, box=True)
                    if bool(obj["mask_present"]) != (point is not None and box is not None):
                        raise ValueError("Mask presence disagrees with geometry")
                    if not obj["mask_present"] and (point is not None or box is not None):
                        raise ValueError("Absent mask cannot supply geometry")
                    if obj["mask_present"] and not obj["mask_path"]:
                        raise ValueError("Present mask requires a source mask artifact")
                    identity = f"{manifest_hash}:{clip['path']}:{object_id}"
                    evidence = {
                        "extraction_sha256": manifest_hash,
                        "clip": clip["path"],
                        "source_frame_sha256": frame["sha256"],
                        "mask_path": obj["mask_path"],
                        "mask_sha256": digest(directory / obj["mask_path"]) if obj["mask_path"] else None,
                        "interval": {"start_frame": clip["start_frame"], "end_frame": clip["end_frame"]},
                    }
                    bucket["detections"].append(
                        {
                            "label": candidate["name"],
                            "object_id": identity,
                            "bbox_format": "xyxy",
                            "bbox": box,
                            "bbox_max_exclusive": True,
                            "point_format": "xy",
                            "point": point,
                            "point_source": "mask_centroid",
                            "seed_point": coordinates(candidate["point"], size) if index == 0 else None,
                            "seed_point_source": candidate.get("point_source", "molmo")
                            if index == 0
                            else None,
                            "seed_review": candidate.get("seed_review") if index == 0 else None,
                            "mask_present": obj["mask_present"],
                            "mask_area_fraction": obj["area_fraction"],
                            "missing_reason": obj.get("missing_reason"),
                            "visibility": "unknown",
                            "review": "pending",
                            "evidence": evidence,
                        }
                    )
                    history[identity].append(
                        {"frame_index": frame["frame_index"], "timestamp": frame["timestamp"], "point": point}
                    )
                    bucket["traces"].append(
                        {
                            "label": candidate["name"],
                            "object_id": identity,
                            "entity": "object",
                            "point_format": "xy",
                            "samples": list(history[identity]),
                            "review": "pending",
                        }
                    )
    return events, provenance


def native_rows(bucket: dict, camera: str) -> list[dict]:
    """Use existing vqa/trace styles and the canonical event row fields, without tool calls."""
    common = {
        "image_size": bucket["image_size"],
        "coordinate_system": "original_image_pixels_x_right_y_down",
        "review": "pending",
        "human_verified": False,
        "accepted_training_labels": False,
    }

    def row(style, role, content):
        return {"role": role, "content": content, "style": style, "camera": camera, "tool_calls": None}

    return [
        row(
            "vqa",
            "user",
            "Report the candidate object boxes, mask-centroid points, and available pointing seeds in this view.",
        ),
        row("vqa", "assistant", json.dumps({**common, "detections": bucket["detections"]}, sort_keys=True)),
        row(
            "trace",
            "assistant",
            json.dumps(
                {**common, "temporal_scope": "observed_prefix_only", "trajectories": bucket["traces"]},
                sort_keys=True,
            ),
        ),
    ]


def event_array(rows: list[list[dict]]) -> pa.Array:
    # Arrow's native JSON extension expects serialized tool calls. HF datasets.Json
    # decodes them again; no existing speech/tool events are discarded.
    serialized = [
        [
            {
                **r,
                "tool_calls": None
                if r.get("tool_calls") is None
                else [call if isinstance(call, str) else json.dumps(call) for call in r["tool_calls"]],
            }
            for r in (frame or [])
        ]
        for frame in rows
    ]
    canonical = language_events_arrow_type()
    storage = pa.list_(
        pa.struct(
            [
                pa.field(
                    f.name, pa.list_(pa.string()) if f.name == "tool_calls" else f.type, nullable=f.nullable
                )
                for f in canonical.value_type
            ]
        )
    )
    return pa.array(serialized, type=storage).cast(canonical)


def export_dataset(dataset: Path, extractions: list[Path], output: Path) -> dict:
    dataset, output = dataset.resolve(), output.resolve()
    if output.exists() or dataset in output.parents:
        raise ValueError("Output must be a fresh directory outside the source dataset")
    source = read_json(dataset / "source.json")
    source = {k: source[k] for k in ("repo_id", "revision")}
    events, provenance = collect_candidates(extractions, source)
    info = read_json(dataset / "meta/info.json")
    by_frame = defaultdict(dict)
    for (episode, frame, camera), bucket in events.items():
        feature = info["features"].get(camera)
        dimensions = dict(zip(feature.get("names") or [], feature["shape"], strict=True)) if feature else {}
        if [dimensions.get("width"), dimensions.get("height")] != bucket["image_size"]:
            raise ValueError(f"Unknown camera or mismatched source image size: {camera}")
        by_frame[(episode, frame)][camera] = bucket
    # Validate all frame identities/timestamps and collisions before creating output.
    shards = sorted((dataset / "data").rglob("*.parquet"))
    seen = set()
    shard_hashes = {}
    for path in shards:
        table = pq.read_table(path)
        shard_hashes[str(path.relative_to(dataset))] = digest(path)
        columns = ["episode_index", "frame_index", "timestamp"]
        if LANGUAGE_EVENTS in table.column_names:
            columns.append(LANGUAGE_EVENTS)
        for frame in table.select(columns).to_pylist():
            key = (frame["episode_index"], frame["frame_index"])
            if key not in by_frame:
                continue
            if key in seen:
                raise ValueError("Duplicate source frame")
            seen.add(key)
            for camera, bucket in by_frame[key].items():
                if frame["timestamp"] != bucket["timestamp"]:
                    raise ValueError("Extraction timestamp does not exactly match source frame")
                if any(
                    r["camera"] == camera and r["style"] in {"vqa", "trace"}
                    for r in (frame.get(LANGUAGE_EVENTS) or [])
                ):
                    raise ValueError(
                        "Existing camera VQA/trace collision; refusing to overwrite or make ambiguous rows"
                    )
    if seen != set(by_frame):
        raise ValueError("Extraction contains frames absent from the source dataset")
    output.mkdir(parents=True)
    shutil.copytree(dataset / "meta", output / "meta")
    shutil.copy2(dataset / "source.json", output / "source.json")
    if (dataset / "videos").exists():
        (output / "videos").symlink_to(dataset / "videos", target_is_directory=True)
    for path in shards:
        if digest(path) != shard_hashes[str(path.relative_to(dataset))]:
            raise ValueError("Source changed during export")
        table = pq.read_table(path)
        old = (
            table[LANGUAGE_EVENTS].to_pylist()
            if LANGUAGE_EVENTS in table.column_names
            else [[] for _ in range(len(table))]
        )
        for i, (episode, frame) in enumerate(
            zip(table["episode_index"].to_pylist(), table["frame_index"].to_pylist(), strict=True)
        ):
            additions = by_frame.get((episode, frame), {})
            if additions:
                old[i] = [
                    *(old[i] or []),
                    *(r for camera, bucket in sorted(additions.items()) for r in native_rows(bucket, camera)),
                ]
        array = event_array(old)
        if LANGUAGE_EVENTS in table.column_names:
            table = table.set_column(table.column_names.index(LANGUAGE_EVENTS), LANGUAGE_EVENTS, array)
        else:
            table = table.append_column(LANGUAGE_EVENTS, array)
        target = output / path.relative_to(dataset)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_table_one_row_group_per_episode(table, target)
    report = {
        "source": source,
        "source_data_sha256": shard_hashes,
        "extractions": provenance,
        "annotated_frames": len(seen),
        "camera_frames": len(events),
        "language_rows_added": 3 * len(events),
        "review": "pending",
        "accepted_training_labels": False,
        "video_reference": str(dataset / "videos"),
        "exporter_sha256": digest(Path(__file__)),
    }
    info["features"][LANGUAGE_EVENTS] = language_feature_info()[LANGUAGE_EVENTS]
    (output / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (output / "meta/grounding_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return {k: report[k] for k in ("annotated_frames", "camera_frames", "language_rows_added", "review")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--extractions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_dataset(args.dataset_root, args.extractions, args.output)))


if __name__ == "__main__":
    main()
