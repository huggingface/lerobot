# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Calibrated native SOMA geometry; no MANO fitting or reference data."""

import numpy as np

COMMON21 = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24])


def verify_joint_layout(asset):
    with np.load(asset, allow_pickle=False) as z:
        for side in ("Left", "Right"):
            expected = [side + "Hand"]
            expected += [side + "HandThumb" + suffix for suffix in ("1", "2", "3", "End")]
            for finger in ("Index", "Middle", "Ring", "Pinky"):
                expected += [side + "Hand" + finger + suffix for suffix in ("1", "2", "3", "4", "End")]
            assert z[side.lower() + "_pose_reference_joint_names"].tolist() == expected


def validate_intrinsics(matrix):
    k = np.asarray(matrix, dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Explicit finite positive camera intrinsics are mandatory")
    if not np.array_equal(k[2], [0, 0, 1]):
        raise ValueError("Invalid pinhole calibration")
    return k


def normalize_hands(result, matrix, variant):
    if variant not in ("raw", "smoothed"):
        raise ValueError(variant)
    n = result["source_frame_count"]
    if result["coordinates"]["geometry"] != "camera_meters":
        raise ValueError("Expected physical-side camera-space geometry in meters")
    joints = np.full((n, 2, 21, 3), np.nan)
    observed = np.zeros((n, 2), dtype=bool)
    predicted = np.zeros((n, 2), dtype=bool)
    seen = set()
    for frame in result["frames"]:
        i = frame["frame_id"]
        if i not in range(n):
            raise ValueError("Invalid frame index")
        for row in frame["instances"]:
            h = ("left", "right").index(row["side"])
            if (i, h) in seen:
                raise ValueError("Multiple predictions for a physical side")
            seen.add((i, h))
            view = row if variant == "raw" else row["smoothed"]
            value = np.asarray(view["joints_camera"], dtype=np.float64)
            if value.shape != (25, 3):
                raise ValueError("Expected native SOMA25")
            # Geometry is already translated and mirrored. Never add translation twice.
            joints[i, h] = value[COMMON21]
            predicted[i, h] = np.isfinite(value).all() and (value[:, 2] > 0).all()
            observed[i, h] = bool(row["bbox_observed"]) and predicted[i, h]
    return {
        "joints_camera": joints,
        "direct_valid": observed,
        "tracked_valid": predicted,
        "source_frame_index": np.arange(n),
        "calibrated_intrinsics": np.broadcast_to(validate_intrinsics(matrix), (n, 3, 3)),
        "representation": np.asarray("native-SOMA-MHR-common21-anatomical-proxy"),
        "variant": np.asarray(variant),
    }
