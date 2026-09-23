# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Compile candidate pick/place commands from native, hash-bound language annotations.

Object roles must be reviewed explicitly. This step never approves training commands,
fills missing masks, changes action targets, or substitutes object paths for gripper traces.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compile_features(root: Path, bindings: dict) -> tuple[dict, dict]:
    source = json.loads((root / "source.json").read_text())
    source = {key: source[key] for key in ("repo_id", "revision")}
    if source != bindings["source"]:
        raise ValueError("Role bindings refer to a different dataset source")
    provenance_path = root / "meta/grounding_provenance.json"
    if digest(provenance_path) != bindings["grounding_provenance_sha256"]:
        raise ValueError("Role bindings refer to stale grounding provenance")
    if json.loads(provenance_path.read_text())["source"] != source:
        raise ValueError("Grounding provenance refers to a different dataset source")
    shards = sorted((root / "data").rglob("*.parquet"))
    actual = {str(path.relative_to(root)): digest(path) for path in shards}
    if not actual or actual != bindings["data_sha256"]:
        raise ValueError("Role bindings refer to stale or incomplete native data shards")
    info = json.loads((root / "meta/info.json").read_text())
    requests = bindings["segments"]
    if not requests:
        raise ValueError("At least one role-bound interval is required")
    wanted = set()
    for segment in requests:
        episode, start, end = (segment[k] for k in ("episode_index", "start_frame", "end_frame"))
        if any(type(v) is not int for v in (episode, start, end)) or episode < 0 or not 0 <= start < end:
            raise ValueError("Invalid requested interval")
        keys = {(episode, frame) for frame in range(start, end)}
        if wanted & keys:
            raise ValueError("Overlapping requested intervals")
        wanted.update(keys)
        if not isinstance(segment.get("subtask"), str) or not segment["subtask"].strip():
            raise ValueError("A source subtask is required")
        if not segment.get("subtask_evidence"):
            raise ValueError("Subtask provenance is required")
        review = segment["role_review"]
        reviewer = review.get("reviewer", {})
        if (
            review.get("verdict") != "accepted"
            or reviewer.get("kind") not in {"model", "human"}
            or not reviewer.get("id", "").strip()
            or not review.get("notes", "").strip()
        ):
            raise ValueError("Pick/place roles require an accepted attributed role review")
        roles = segment["objects"]
        if [obj["role"] for obj in roles] != ["pick", "place"]:
            raise ValueError("Objects must be ordered pick then place")
        if roles[0]["object_id"] == roles[1]["object_id"]:
            raise ValueError("Pick and place roles must identify distinct objects")
        if len(segment["image_size"]) != 2 or any(
            type(v) is not int or v <= 0 for v in segment["image_size"]
        ):
            raise ValueError("Image size must contain two positive integer dimensions")
        feature = info["features"].get(segment["camera"], {})
        dimensions = dict(zip(feature.get("names", []), feature.get("shape", []), strict=True))
        if [dimensions.get("width"), dimensions.get("height")] != segment["image_size"]:
            raise ValueError("Binding camera dimensions differ from dataset metadata")

    frames = {}
    episodes = sorted({episode for episode, _ in wanted})
    for path in shards:
        table = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "timestamp", "language_events"],
            filters=[("episode_index", "in", episodes)],
        )
        for row in table.to_pylist():
            key = (row["episode_index"], row["frame_index"])
            if key in wanted:
                if key in frames:
                    raise ValueError("Duplicate requested source frame")
                frames[key] = row
    if set(frames) != wanted:
        raise ValueError("Requested intervals contain absent source frames")

    features, reports = [], []
    for segment in requests:
        episode, start, end = (segment[k] for k in ("episode_index", "start_frame", "end_frame"))
        camera, size = segment["camera"], segment["image_size"]
        geometry, missing = {}, []
        for index in range(start, end):
            row = frames[(episode, index)]
            answers = [
                event
                for event in (row["language_events"] or [])
                if event["camera"] == camera and event["style"] == "vqa" and event["role"] == "assistant"
            ]
            if not answers:
                missing.append({"frame_index": index, "reason": "missing_camera_annotation"})
                continue
            if len(answers) != 1:
                raise ValueError("Ambiguous native camera VQA answers")
            answer = json.loads(answers[0]["content"])
            if (
                answer["image_size"] != size
                or answer["coordinate_system"] != "original_image_pixels_x_right_y_down"
            ):
                raise ValueError("Native annotation coordinate frame differs from bindings")
            points = []
            for role in segment["objects"]:
                matches = [obj for obj in answer["detections"] if obj["object_id"] == role["object_id"]]
                if not matches:
                    missing.append({"frame_index": index, "role": role["role"], "reason": "missing_identity"})
                    continue
                if len(matches) != 1:
                    raise ValueError("Duplicate native object identity")
                obj = matches[0]
                if (
                    obj["label"] != role["name"]
                    or obj["entity"] != "object"
                    or obj["point_source"] != "mask_centroid"
                ):
                    raise ValueError("Native object identity or point source differs from reviewed binding")
                if not obj["mask_present"]:
                    if obj["point"] is not None or obj["bbox"] is not None:
                        raise ValueError("Missing mask has invented geometry")
                    missing.append({"frame_index": index, "role": role["role"], "reason": "missing_mask"})
                    continue
                point = obj["point"]
                if (
                    not isinstance(point, list)
                    or len(point) != 2
                    or any(
                        isinstance(v, bool)
                        or not isinstance(v, (int, float))
                        or not math.isfinite(v)
                        or not 0 <= v < limit
                        for v, limit in zip(point, size, strict=True)
                    )
                ):
                    raise ValueError("Invalid native mask centroid")
                # Rounding can reach width/height for subpixel centers near an edge.
                points.append([min(limit - 1, round(v)) for v, limit in zip(point, size, strict=True)])
            if len(points) == 2:
                geometry[str(index)] = points
        reports.append(
            {
                "episode_index": episode,
                "start_frame": start,
                "end_frame": end,
                "status": "missing_geometry" if missing else "candidate_ready_for_command_review",
                "missing": missing,
            }
        )
        if missing:
            continue  # Keep the full requested interval unresolved; never silently shorten its action horizon.
        evidence = {
            "method": "rounded per-frame native SAM2 mask centroids",
            "grounding_provenance_sha256": bindings["grounding_provenance_sha256"],
            "data_sha256": actual,
            "objects_in_order": segment["objects"],
            "role_review": segment["role_review"],
            "review": "pending",
            "accepted_training_labels": False,
        }
        features.append(
            {
                **{
                    key: segment[key]
                    for key in ("episode_index", "start_frame", "end_frame", "subtask", "subtask_evidence")
                },
                "views": [
                    {
                        "camera": camera,
                        "image_size": size,
                        "targets": [
                            {
                                "instruction": "pick up the object at the first point and place it into the target at the second point",
                                "points_by_frame": geometry,
                                "evidence": evidence,
                            }
                        ],
                    }
                ],
            }
        )
    if any(digest(root / name) != value for name, value in actual.items()):
        raise ValueError("Native data changed during compilation")
    common = {
        "source": source,
        "review": "pending",
        "human_verified": False,
        "accepted_training_labels": False,
    }
    return ({**common, "segments": features}, {**common, "segments": reports})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.dataset_root.resolve() in args.output.resolve().parents:
        raise ValueError("Use a fresh output outside the dataset")
    bindings = json.loads(args.bindings.read_text())
    features, report = compile_features(args.dataset_root, bindings)
    args.output.mkdir(parents=True)
    for name, value in (
        ("features.candidates.json", features),
        ("coverage.json", report),
        ("bindings.json", bindings),
    ):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {"candidate_segments": len(features["segments"]), "requested_segments": len(report["segments"])}
        )
    )


if __name__ == "__main__":
    main()
