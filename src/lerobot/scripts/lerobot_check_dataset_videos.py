#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Check local LeRobot v3 video files against their episode metadata."""

from __future__ import annotations

import argparse
import json
import math
import operator
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def probe_video(path: Path) -> dict[str, Any]:
    """Read a video's stream timing metadata without decoding every frame."""
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "PyAV is required; install the dataset extra with `pip install 'lerobot[dataset]'`."
        ) from exc

    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("No video stream")
            stream = container.streams.video[0]
            if stream.time_base is None or stream.time_base <= 0:
                raise ValueError("Video stream has an invalid time base")
            return {
                "duration_s": (
                    float(stream.duration * stream.time_base) if stream.duration is not None else None
                ),
                "time_base_s": float(stream.time_base),
                "start_time_s": float((stream.start_time or 0) * stream.time_base),
                "declared_frames": stream.frames,
            }
    except av.error.FFmpegError as exc:
        raise ValueError(str(exc)) from exc


def _metadata_index(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def inspect_dataset(root: Path, *, tolerance_s: float = 1e-4) -> dict[str, Any]:
    """Check local v3.0 episode intervals against the referenced video streams.

    The check reads ``meta/info.json``, episode Parquet metadata, and video
    container headers. It never downloads or modifies files and does not
    decode all frames, so it cannot detect corruption confined to frame data.
    """
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be finite and nonnegative")
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")

    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read dataset metadata at {info_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {info_path}: {exc}") from exc

    if not isinstance(info, dict):
        raise ValueError(f"Expected an object in {info_path}")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Only local LeRobot v3.0 datasets are supported")
    features = info.get("features")
    template = info.get("video_path")
    if not isinstance(features, dict):
        raise ValueError("meta/info.json must contain a features object")
    video_keys = [
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    if not video_keys:
        return {
            "ok": True,
            "scope": "container metadata only; video frames are not fully decoded",
            "root": str(root),
            "tolerance_s": tolerance_s,
            "episodes_checked": 0,
            "intervals_checked": 0,
            "unique_videos": 0,
            "videos": {},
            "issues": [],
            "message": "No video features found",
        }
    if not isinstance(template, str) or not template:
        raise ValueError("A nonempty video_path template is required for video features")

    metadata_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not metadata_files:
        raise ValueError(f"No episode Parquet files found under {root / 'meta' / 'episodes'}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "PyArrow is required; install the dataset extra with `pip install 'lerobot[dataset]'`."
        ) from exc

    report: dict[str, Any] = {
        "ok": False,
        "scope": "container metadata only; video frames are not fully decoded",
        "root": str(root),
        "tolerance_s": tolerance_s,
        "episodes_checked": 0,
        "intervals_checked": 0,
        "unique_videos": 0,
        "videos": {},
        "issues": [],
    }

    def add_issue(code: str, episode: int, key: str, message: str, path: str | None = None) -> None:
        issue = {"code": code, "episode_index": episode, "video_key": key, "message": message}
        if path is not None:
            issue["path"] = path
        report["issues"].append(issue)

    for metadata_file in metadata_files:
        columns = ["episode_index"]
        for key in video_keys:
            columns.extend(
                f"videos/{key}/{field}"
                for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
            )
        try:
            parquet = pq.ParquetFile(metadata_file)
            missing_columns = sorted(set(columns) - set(parquet.schema_arrow.names))
            if missing_columns:
                raise ValueError(f"missing columns {missing_columns}")
            batches = parquet.iter_batches(batch_size=1024, columns=columns)
            for batch in batches:
                for row in batch.to_pylist():
                    episode = _metadata_index(row["episode_index"], "episode_index")
                    report["episodes_checked"] += 1
                    for key in video_keys:
                        report["intervals_checked"] += 1
                        prefix = f"videos/{key}/"
                        try:
                            start = float(row[prefix + "from_timestamp"])
                            end = float(row[prefix + "to_timestamp"])
                            chunk = _metadata_index(row[prefix + "chunk_index"], prefix + "chunk_index")
                            file = _metadata_index(row[prefix + "file_index"], prefix + "file_index")
                        except (TypeError, ValueError, OverflowError) as exc:
                            add_issue("invalid_metadata", episode, key, str(exc))
                            continue

                        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                            add_issue("invalid_interval", episode, key, f"Invalid interval [{start}, {end})")
                            continue
                        try:
                            relative_path = template.format(video_key=key, chunk_index=chunk, file_index=file)
                            video_path = (root / relative_path).resolve()
                        except (IndexError, KeyError, ValueError, OSError) as exc:
                            raise ValueError(f"Unsupported video_path template {template!r}: {exc}") from exc
                        if not video_path.is_relative_to(root):
                            raise ValueError(f"video_path points outside the dataset root: {video_path}")
                        display_path = video_path.relative_to(root).as_posix()

                        if display_path not in report["videos"]:
                            if not video_path.is_file():
                                probe = {"error_code": "missing_video", "error": "File does not exist"}
                            else:
                                try:
                                    probe = probe_video(video_path)
                                except (OSError, ValueError) as exc:
                                    probe = {"error_code": "unreadable_video", "error": str(exc)}
                            report["videos"][display_path] = probe

                        probe = report["videos"][display_path]
                        if "error_code" in probe:
                            add_issue(probe["error_code"], episode, key, probe["error"], display_path)
                        elif probe["duration_s"] is None:
                            add_issue(
                                "unknown_duration", episode, key, "Video stream has no duration", display_path
                            )
                        elif abs(probe["start_time_s"]) > tolerance_s:
                            add_issue(
                                "unsupported_timeline",
                                episode,
                                key,
                                "Video stream starts at a nonzero timestamp",
                                display_path,
                            )
                        else:
                            # A whole-frame tolerance would hide the missing final frame this check targets.
                            if end > probe["duration_s"] + tolerance_s:
                                add_issue(
                                    "interval_past_video_end",
                                    episode,
                                    key,
                                    f"Episode end {end:.9f}s exceeds video duration "
                                    f"{probe['duration_s']:.9f}s (tolerance {tolerance_s:.9f}s)",
                                    display_path,
                                )
        except ValueError as exc:
            raise ValueError(f"Invalid episode metadata in {metadata_file}: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Cannot read episode metadata in {metadata_file}: {exc}") from exc

    if report["episodes_checked"] == 0:
        raise ValueError("Episode metadata is empty")
    report["unique_videos"] = len(report["videos"])
    report["ok"] = not report["issues"]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local video-integrity check and print a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Local LeRobot v3.0 dataset root; nothing is downloaded")
    parser.add_argument("--tolerance-s", type=float, default=1e-4, help="Timestamp tolerance in seconds")
    args = parser.parse_args(argv)
    try:
        report = inspect_dataset(args.root, tolerance_s=args.tolerance_s)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(json.dumps({"ok": False, "fatal_error": str(exc)}, ensure_ascii=False), file=sys.stdout)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
