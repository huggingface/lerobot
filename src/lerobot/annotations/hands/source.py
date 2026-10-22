# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Content-bound raw-MP4 discovery, without constructing a GPU runtime."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

from ..runtime.artifacts import file_checksum, local_file_token
from ..runtime.types import Artifact, DatasetRef, InputItem, fingerprint


def probe_video(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("Install ffmpeg/ffprobe before inspecting raw videos")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
    fps = numerator / denominator if denominator else 0
    duration = float(stream.get("duration", payload.get("format", {}).get("duration", 0)))
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid video clock: {path}")
    frames = stream.get("nb_frames")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "seconds": duration,
        "frames": int(frames) if str(frames).isdigit() else round(fps * duration),
    }


def validate_calibration(path: Path, video: dict) -> None:
    calibration = json.loads(path.read_text())
    if calibration.get("model") not in {"pinhole", "opencv_fisheye"}:
        raise ValueError(f"Supply pinhole or opencv_fisheye calibration: {path}")
    if any(calibration.get(k) != video[k] for k in ("width", "height")):
        raise ValueError(f"Calibration resolution differs from the video: {path}")
    for key in ("fx", "fy", "cx", "cy"):
        value = calibration.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"Calibration requires finite {key}: {path}")
    if calibration["fx"] <= 0 or calibration["fy"] <= 0:
        raise ValueError(f"Calibration focal lengths must be positive: {path}")
    # The selected calibrated recipe deliberately rejects anisotropic K instead
    # of silently falling back to the packaged approximate camera model.
    if not math.isclose(calibration["fx"], calibration["fy"], rel_tol=1e-6):
        raise ValueError(f"The calibrated recipe requires isotropic fx/fy: {path}")


class RawHandVideoSource:
    def __init__(self, root: Path, videos_per_group: int = 32):
        self.root = root.resolve()
        self.files: dict[str, Path] = {}
        self.source_tokens: dict[Path, tuple] = {}
        self.videos = []
        paths = sorted(p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4")
        if not paths:
            raise ValueError("No MP4 videos found")
        for index, path in enumerate(paths):
            calibration = path.with_suffix(".intrinsics.json")
            if not calibration.is_file():
                raise ValueError(f"Missing per-video calibration: {calibration}")
            metadata = probe_video(path)
            validate_calibration(calibration, metadata)

            def artifact(file, suffix):
                token = local_file_token(file)
                digest, size = file_checksum(file)
                if local_file_token(file) != token:
                    raise ValueError("Source changed during discovery")
                self.source_tokens[file] = token
                relative = f"hand-inputs/{digest}{suffix}"
                self.files[relative] = file
                return {"path": relative, "sha256": digest, "size": size}

            self.videos.append(
                {
                    "source_video_id": index,
                    "relative_path": path.relative_to(self.root).as_posix(),
                    **metadata,
                    "video": artifact(path, ".mp4"),
                    "calibration": artifact(calibration, ".json"),
                }
            )
        self.groups = [
            self.videos[i : i + videos_per_group] for i in range(0, len(self.videos), videos_per_group)
        ]
        first = self.videos[0]
        if any(
            v["width"] != first["width"]
            or v["height"] != first["height"]
            or not math.isclose(v["fps"], first["fps"], rel_tol=1e-6)
            for v in self.videos
        ):
            raise ValueError(
                "One output dataset requires matching FPS and resolution; split the input folders"
            )
        self.dataset_ref = DatasetRef("raw-mp4:" + str(self.root), fingerprint(self.videos))

    def discover(self, stage, store, upstream):
        for index, videos in enumerate(self.groups):
            yield InputItem(
                f"group-{index:06d}",
                {"videos": videos},
                cost=sum(v["frames"] for v in videos),
                physical_seconds=sum(v["seconds"] for v in videos),
                content_digest=fingerprint(videos),
            )

    def stage_inputs(self, store, plan):
        # Called only before execution. --plan neither uploads raw media nor
        # initializes a model. All workers consume the same checksum-bound bytes.
        if any(local_file_token(path) != token for path, token in self.source_tokens.items()):
            raise ValueError("Source changed after discovery; create a fresh plan")
        for video in self.videos:
            for kind in ("video", "calibration"):
                artifact = Artifact(**video[kind])
                if store.exists(artifact.path):
                    if not store.verify(artifact):
                        raise ValueError("Existing hand input checksum mismatch")
                else:
                    actual = store.put_file(artifact.path, self.files[artifact.path])
                    if (actual.sha256, actual.size) != (artifact.sha256, artifact.size):
                        raise ValueError("Source changed after discovery; create a fresh plan")
