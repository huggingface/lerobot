# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Explicit experimental MoGe focal estimation; never a measured lens calibration."""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

from .assets import file_checksum
from .config import HandBackendConfig

METHOD = "moge-v2-fov-median-v1"


def summarize_intrinsics(matrices: list, frames: list[int], width: int, height: int) -> dict:
    """Aggregate normalized pinhole predictions without inventing lens distortion."""
    if min(width, height) < 1:
        raise ValueError("Intrinsics need a positive source resolution")
    values = np.asarray(matrices, dtype=np.float64)
    if values.shape != (len(frames), 3, 3) or not frames:
        raise ValueError("MoGe must predict intrinsics for every sampled frame")
    if not np.isfinite(values).all() or np.any(values[:, (0, 1), (0, 1)] <= 0):
        raise ValueError("MoGe predicted invalid focal lengths")
    if (
        not np.allclose(values[:, 2], [0, 0, 1])
        or not np.allclose(values[:, :2, 2], 0.5)
        or not np.allclose(values[:, (0, 1), (1, 0)], 0)
    ):
        raise ValueError("Expected MoGe normalized intrinsics with a centered principal point")
    focal = values[:, (0, 1), (0, 1)] * [width, height]
    # The upstream model assumes square pixels; do not silently accept a changed convention.
    if not np.allclose(focal[:, 0], focal[:, 1], rtol=1e-4):
        raise ValueError("MoGe focal output does not match the source aspect ratio")
    median = np.median(focal, axis=0)
    return {
        "model": "pinhole",
        "width": width,
        "height": height,
        "fx": float(median[0]),
        "fy": float(median[1]),
        "cx": width / 2,
        "cy": height / 2,
        "distortion": [],
        "source": "moge-estimated",
        "method": METHOD,
        "measured": False,
        "assumptions": ["centered principal point", "square pixels", "no lens distortion"],
        "sample_frames": frames,
        "focal_samples_px": focal.tolist(),
        "relative_focal_mad": (np.median(np.abs(focal - median), axis=0) / median).tolist(),
        "metric_accuracy_validated": False,
    }


def resolve_calibration(calibration: dict, video: Path, directory: Path, config: HandBackendConfig) -> dict:
    """Run the pinned camera interpreter only for an explicitly selected estimate."""
    from .writer import validate_calibration

    if calibration.get("method") != METHOD or "fx" in calibration:
        return calibration
    destination = directory / "estimated-intrinsics.json"
    subprocess.run(
        [config.droid_python, "-m", __name__, str(video), config.droid_moge_checkpoint, str(destination)],
        check=True,
        timeout=config.timeout_s,
    )
    result = json.loads(destination.read_text())
    if result.get("method") != METHOD or result.get("measured") is not False:
        raise ValueError("The intrinsics worker did not report an explicit MoGe estimate")
    return validate_calibration(result, calibration["width"], calibration["height"])


def estimate_video(video: Path, checkpoint: Path, destination: Path):
    """Use the existing MoGe Small checkpoint, with no FOV input or new model downloads."""
    import cv2
    import torch
    from moge.model.v2 import MoGeModel

    capture = cv2.VideoCapture(str(video))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or min(width, height, count) < 1:
            raise ValueError("Cannot read the video for MoGe intrinsics estimation")
        frames = np.unique(np.rint(np.linspace(0, count - 1, min(count, 5))).astype(int)).tolist()
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=r".*parameter\(s\).*")
            model = MoGeModel.from_pretrained(str(checkpoint)).to("cuda").eval()
        if not hasattr(model, "scale_head"):
            raise ValueError("Use the pinned MoGe v2 Small metric checkpoint")
        matrices = []
        for index in frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok or image.shape[:2] != (height, width):
                raise ValueError(f"Cannot decode intrinsics sample frame {index}")
            # Preserve the aspect ratio. Intrinsics returned by MoGe are normalized to this image.
            scale = min(1.0, 720 / max(width, height))
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(image).permute(2, 0, 1).float().to("cuda") / 255
            with torch.inference_mode():
                prediction = model.infer(tensor, fov_x=None, num_tokens=1200, use_fp16=False)
            matrix = prediction["intrinsics"].detach().cpu().numpy()
            # Account for rounding in the small inference image before restoring square source pixels.
            matrix[0, 0] *= size[0] / size[1] / (width / height)
            matrices.append(matrix.tolist())
        report = summarize_intrinsics(matrices, frames, width, height)
        report.update(checkpoint_sha256=file_checksum(checkpoint)[0], num_tokens=1200, use_fp16=False)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        capture.release()


if __name__ == "__main__":
    estimate_video(*(Path(value) for value in sys.argv[1:]))
