# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Attach native SOMA156 actions without dropping source rows or inventing poses."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyarrow as pa

from .config import ACTION_SCHEMA
from .egohand.output import NATIVE_FIELDS

ACTION_NAMES = [
    f"{hand}_{field}_{axis}" for hand in ("a", "b") for field in ("pos", "rot") for axis in "xyz"
] + [
    f"{hand}_soma_joint_{joint:02d}_rot_{axis}"
    for hand in ("a", "b")
    for joint in range(1, 25)
    for axis in "xyz"
]
PARAMETERS = {
    **NATIVE_FIELDS,
    "hand_joints_camera": (2, 21, 3),
    "hand_joints_world": (2, 21, 3),
    "camera_world_from_optical": (4, 4),
}
FEATURES = {
    "action": {"dtype": "float32", "shape": [156], "names": ACTION_NAMES},
    "annotation.hand_action_valid": {"dtype": "bool", "shape": [1], "names": ["valid"]},
    "annotation.hand_action_observed_valid": {"dtype": "bool", "shape": [1], "names": ["observed_valid"]},
    "annotation.hand_observed": {"dtype": "bool", "shape": [2], "names": ["left", "right"]},
    "annotation.hand_reference_frame": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
    **{
        "annotation." + name: {
            "dtype": "float32",
            "shape": [math.prod(shape)],
            "names": [f"{name}_{i}" for i in range(math.prod(shape))],
        }
        for name, shape in PARAMETERS.items()
    },
}


def validate_calibration(calibration: dict, width: int, height: int) -> dict:
    if calibration.get("model") not in {"pinhole", "opencv_fisheye"}:
        raise ValueError(
            "Hand actions require measured camera.intrinsics in the selected camera's dataset metadata"
        )
    if calibration.get("width") != width or calibration.get("height") != height:
        raise ValueError("Hand calibration resolution must match the selected camera")
    for key in ("fx", "fy", "cx", "cy"):
        value = calibration.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Hand calibration requires finite {key}")
    if min(calibration["fx"], calibration["fy"]) <= 0:
        raise ValueError("Camera focal lengths must be positive")
    coefficients = calibration.get("distortion", [])
    if not isinstance(coefficients, list) or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in coefficients
    ):
        raise ValueError("Invalid camera distortion coefficients")
    if calibration["model"] == "opencv_fisheye" and len(coefficients) != 4:
        raise ValueError("Fisheye calibration requires four distortion coefficients")
    if calibration["model"] == "pinhole" and len(coefficients) not in {0, 4, 5, 8, 12, 14}:
        raise ValueError("Unsupported pinhole distortion coefficient count")
    return calibration


def attach_actions(table: pa.Table, report: dict, *, directory: Path, fps: float) -> tuple[pa.Table, dict]:
    n = len(table)
    if table["frame_index"].to_pylist() != list(range(n)):
        raise ValueError("Hand annotation requires ordered source episode frames")
    if report.get("action_schema") != ACTION_SCHEMA or report.get("gt_scale_alignment") is not False:
        raise ValueError("Expected native SOMA156 actions without GT scale alignment")
    if report.get("motion_infilling") is not False:
        raise ValueError("The validated EgoHand recipe does not synthesize missing hand poses")
    if report["rectified_frames"] != n or not math.isclose(report["rectified_fps"], fps, rel_tol=1e-6):
        raise ValueError("Native hand frame count/rate differs from the source episode")
    if "annotation.original_action" in table.column_names:
        raise ValueError("This dataset already has preserved hand-action source data")
    timestamps = table["timestamp"].to_numpy()
    source_stamps = np.asarray(timestamps, dtype=np.float64)
    tolerance = max(1e-5, 2 * float(np.abs(np.spacing(timestamps)).max()))
    if tolerance >= 0.25 / fps:
        raise ValueError("Source timestamp precision is too coarse to align hand actions")
    actions = np.zeros((n, 156), dtype=np.float32)
    parameters = {
        name: np.zeros((n, math.prod(shape)), dtype=np.float32) for name, shape in PARAMETERS.items()
    }
    valid = np.zeros(n, dtype=bool)
    reference = np.full(n, -1, dtype=np.int64)
    spans = report.get("segments", [])
    for text in spans:
        path = Path(text).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Native action segment escaped its episode workspace")
        with np.load(path, allow_pickle=False) as segment:
            indices, values, stamps = segment["source_frame_index"], segment["action"], segment["timestamp_s"]
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or len(indices) < 30:
                raise ValueError("Native actions require at least 30 contiguous observed bimanual frames")
            if (
                np.any(indices < 0)
                or np.any(indices >= n)
                or np.any(np.diff(indices) != 1)
                or np.any(valid[indices])
            ):
                raise ValueError("Native actions overlap or do not align with source frames")
            if values.shape != (len(indices), 156) or not np.isfinite(values).all():
                raise ValueError("Invalid native SOMA156 hand actions")
            if stamps.shape != (len(indices),) or not np.allclose(
                stamps, source_stamps[indices] - source_stamps[0], atol=tolerance, rtol=0
            ):
                raise ValueError("Native action timestamps differ from source rows")
            for name, shape in PARAMETERS.items():
                value = segment[name]
                if value.shape != (len(indices), *shape) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid native SOMA field: {name}")
                parameters[name][indices] = value.reshape(len(indices), -1)
        actions[indices], valid[indices], reference[indices] = values, True, indices[0]
    if int(valid.sum()) != report["accepted_frames"]:
        raise ValueError("Native hand report and aligned frame counts differ")
    columns = {
        "action": (actions, pa.float32()),
        "annotation.hand_action_valid": (valid, pa.bool_()),
        "annotation.hand_action_observed_valid": (valid, pa.bool_()),
        "annotation.hand_observed": (np.repeat(valid[:, None], 2, axis=1), pa.bool_()),
        "annotation.hand_reference_frame": (reference, pa.int64()),
        **{"annotation." + name: (value, pa.float32()) for name, value in parameters.items()},
    }
    if "action" in table.column_names:
        table = table.append_column("annotation.original_action", table["action"])
    for name, (array, dtype) in columns.items():
        values = pa.array(array.tolist(), type=pa.list_(dtype, array.shape[1]) if array.ndim > 1 else dtype)
        index = table.schema.get_field_index(name)
        table = table.set_column(index, name, values) if index >= 0 else table.append_column(name, values)
    return table, {
        "frames": n,
        "valid_frames": int(valid.sum()),
        "observed_training_frames": int(valid.sum()),
        "infilled_frames": 0,
        "unmeasured_frames": int((~valid).sum()),
        "segments": len(spans),
        "action_schema": ACTION_SCHEMA,
        "action_dimension": 156,
        **({"camera_intrinsics": report["camera_intrinsics"]} if "camera_intrinsics" in report else {}),
    }
