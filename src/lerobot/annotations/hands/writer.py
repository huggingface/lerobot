# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Attach native actions to source rows, retaining every frame and episode."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyarrow as pa

ACTION_NAMES = [
    f"{hand}_{field}_{axis}"
    for hand in ("left", "right")
    for field in ("position", "rotation")
    for axis in "xyz"
] + [
    f"{hand}_finger_joint_{joint:02d}_{axis}"
    for hand in ("left", "right")
    for joint in range(1, 16)
    for axis in "xyz"
]
FEATURES = {
    "action": {"dtype": "float32", "shape": [102], "names": ACTION_NAMES},
    "annotation.hand_action_valid": {"dtype": "bool", "shape": [1], "names": ["valid"]},
    "annotation.hand_action_observed_valid": {"dtype": "bool", "shape": [1], "names": ["observed_valid"]},
    "annotation.hand_observed": {"dtype": "bool", "shape": [2], "names": ["left", "right"]},
    "annotation.hand_infilled": {"dtype": "bool", "shape": [2], "names": ["left", "right"]},
    "annotation.hand_confidence": {"dtype": "float32", "shape": [2], "names": ["left", "right"]},
    "annotation.hand_reference_frame": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
}


def validate_calibration(calibration: dict, width: int, height: int) -> dict:
    if calibration.get("model") not in {"pinhole", "opencv_fisheye"}:
        raise ValueError("Supply measured pinhole or opencv_fisheye camera calibration")
    if calibration.get("width") != width or calibration.get("height") != height:
        raise ValueError("Hand calibration resolution must match the selected camera")
    for key in ("fx", "fy", "cx", "cy"):
        value = calibration.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Hand calibration requires finite {key}")
    if calibration["fx"] <= 0 or not math.isclose(calibration["fx"], calibration["fy"], rel_tol=1e-6):
        raise ValueError("The calibrated hand recipe requires positive, isotropic fx/fy")
    coefficients = calibration.get("distortion", [])
    if not isinstance(coefficients, list) or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in coefficients
    ):
        raise ValueError("Invalid camera distortion coefficients")
    if calibration["model"] == "opencv_fisheye" and len(coefficients) != 4:
        raise ValueError("Fisheye calibration requires four distortion coefficients")
    return calibration


def attach_actions(table: pa.Table, report: dict, *, directory: Path, fps: float) -> tuple[pa.Table, dict]:
    n = len(table)
    if table["frame_index"].to_pylist() != list(range(n)):
        raise ValueError("Hand annotation requires ordered source episode frames")
    if report["rectified_frames"] != n or not math.isclose(report["rectified_fps"], fps, rel_tol=1e-6):
        raise ValueError("Native hand frame count/rate differs from the source episode")
    timestamps = table["timestamp"].to_numpy()
    source_stamps = np.asarray(timestamps, dtype=np.float64)
    # LeRobot timestamps may be float32. Account for their representable clock
    # precision (and subtraction of a nonzero origin) on long recordings.
    timestamp_tolerance = max(1e-5, 2 * float(np.abs(np.spacing(timestamps)).max()))
    if timestamp_tolerance >= 0.25 / fps:
        raise ValueError("Source timestamp precision is too coarse to align hand actions")
    actions = np.zeros((n, 102), dtype=np.float32)
    confidence = np.zeros((n, 2), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    observed = np.zeros((n, 2), dtype=bool)
    infilled = np.zeros((n, 2), dtype=bool)
    observed_valid = np.zeros(n, dtype=bool)
    reference = np.full(n, -1, dtype=np.int64)
    dense = "dense_segments" in report
    spans = report.get("dense_segments", report.get("segments", []))
    for text in spans:
        path = Path(text).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Native action segment escaped its episode workspace")
        with np.load(path, allow_pickle=False) as segment:
            indices = segment["source_frame_index"]
            values = segment["action"]
            scores = segment["confidence"]
            stamps = segment["timestamp_s"]
            direct = segment["observed"] if dense else np.ones((len(indices), 2), dtype=bool)
            filled = segment["infilled"] if dense else np.zeros((len(indices), 2), dtype=bool)
            strict = segment["observed_valid"] if dense else np.ones(len(indices), dtype=bool)
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or len(indices) < (1 if dense else 30)
        ):
            raise ValueError("Invalid native source frame indexes")
        if (
            np.any(indices < 0)
            or np.any(indices >= n)
            or np.any(np.diff(indices) != 1)
            or np.any(valid[indices])
        ):
            raise ValueError("Native actions overlap or do not align with source frames")
        if values.shape != (len(indices), 102) or not np.isfinite(values).all():
            raise ValueError("Invalid native 102D hand actions")
        if scores.shape != (len(indices), 2) or not np.isfinite(scores).all() or np.any(scores <= 0):
            raise ValueError("Hand actions require two finite, positively scored hand predictions")
        if (
            direct.shape != (len(indices), 2)
            or filled.shape != direct.shape
            or strict.shape != (len(indices),)
        ):
            raise ValueError("Invalid hand observation/infilling provenance")
        if (
            any(mask.dtype != np.bool_ for mask in (direct, filled, strict))
            or np.any(direct & filled)
            or not (direct | filled).all()
        ):
            raise ValueError("Every available hand must be labeled observed or infilled")
        if np.any(strict & (~direct.all(axis=1) | filled.any(axis=1))):
            raise ValueError("Infilled predictions cannot be marked as observed training actions")
        relative_stamps = source_stamps[indices] - source_stamps[0]
        if np.shape(stamps) != (len(indices),) or not np.allclose(
            stamps, relative_stamps, atol=timestamp_tolerance, rtol=0
        ):
            raise ValueError("Native action timestamps differ from source rows")
        actions[indices], confidence[indices], valid[indices], reference[indices] = (
            values,
            scores,
            True,
            indices[0],
        )
        observed[indices], infilled[indices], observed_valid[indices] = direct, filled, strict
    if (
        int(valid.sum()) != report.get("recoverable_frames", report["accepted_frames"])
        or int(observed_valid.sum()) != report["accepted_frames"]
    ):
        raise ValueError("Native hand report and aligned frame counts differ")
    columns = {
        "action": (actions, pa.float32()),
        "annotation.hand_action_valid": (valid, pa.bool_()),
        "annotation.hand_action_observed_valid": (observed_valid, pa.bool_()),
        "annotation.hand_observed": (observed, pa.bool_()),
        "annotation.hand_infilled": (infilled, pa.bool_()),
        "annotation.hand_confidence": (confidence, pa.float32()),
        "annotation.hand_reference_frame": (reference, pa.int64()),
    }
    if "annotation.original_action" in table.column_names:
        raise ValueError("This dataset already has preserved hand-action source data")
    if "action" in table.column_names:
        table = table.append_column("annotation.original_action", table["action"])
    for name, (values, dtype) in columns.items():
        array = pa.array(values.tolist(), type=pa.list_(dtype, values.shape[1]) if values.ndim > 1 else dtype)
        index = table.schema.get_field_index(name)
        table = table.set_column(index, name, array) if index >= 0 else table.append_column(name, array)
    return table, {
        "frames": n,
        "valid_frames": int(valid.sum()),
        "observed_training_frames": int(observed_valid.sum()),
        "infilled_frames": int(infilled.any(axis=1).sum()),
        "unmeasured_frames": int((~valid).sum()),
        "segments": len(spans),
    }
