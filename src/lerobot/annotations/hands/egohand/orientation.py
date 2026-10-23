# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Explicit Aria Gen1 sensor-to-upright rotation, independent of GT.

Reference: Project Aria image utilities, rotate_camera_calib_cw90deg.
This is an opt-in dataset adapter, never inferred from hand annotations.
"""

from copy import deepcopy

import numpy as np

UPRIGHT_FROM_SENSOR = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def upright_intrinsics(matrix, height):
    from .adapter import validate_intrinsics

    k = validate_intrinsics(matrix)
    if k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError("This calibrated adapter requires zero-skew pinhole intrinsics")
    pixel_rotation = np.array([[0.0, -1.0, height - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return pixel_rotation @ k @ UPRIGHT_FROM_SENSOR.T


def sensor_pixels(points, height):
    xy = np.asarray(points)
    return np.stack((xy[..., 1], height - 1 - xy[..., 0]), axis=-1)


class UprightFrames:
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return np.ascontiguousarray(np.rot90(self.frames[index], k=3))


def restore_annotations(annotations, matrix, height):
    result = deepcopy(annotations)
    for frame in result.values():
        frame["intrinsics"] = np.asarray(matrix).tolist()
        for instance in frame["instances"]:
            x, y, size = instance["bbox"]
            instance["bbox"] = [y, height - 1 - x, size]
    return result


def restore_predictions(result, geometry, matrix, width, height):
    """Undo camera rotation once, retaining physical-side SOMA parameters."""
    import torch

    output = deepcopy(result)
    rows = []
    for frame in output["frames"]:
        frame["intrinsics"] = torch.as_tensor(matrix, dtype=torch.float32)
        frame["image_size"] = [width, height]
        for row in frame["instances"]:
            box = row["bbox"]
            row["bbox"] = torch.stack((box[1], height - 1 - box[0], box[2]))
            row["intrinsics"] = torch.as_tensor(matrix, dtype=torch.float32)
            for view in (row, row.get("smoothed")):
                if view is None:
                    continue
                rotation = view["joints_camera"].new_tensor(UPRIGHT_FROM_SENSOR)
                for key in ("joints_camera", "vertices_camera", "camera_translation"):
                    view[key] = view[key] @ rotation
                xy = view["keypoints_2d"]
                view["keypoints_2d"] = torch.stack((xy[..., 1], height - 1 - xy[..., 0]), dim=-1)
                view["keypoints_crop"] = (view["keypoints_2d"] - row["bbox"][None, :2]) / row["bbox"][2]
                rows.append((row["side"], view))
    if rows:
        device = geometry._device_anchor.device
        wrist = torch.stack([view["parameters"]["wrist_orient"] for _, view in rows]).to(device)
        right = torch.tensor([side == "right" for side, _ in rows], dtype=torch.bool, device=device)
        camera_rotation = torch.as_tensor(UPRIGHT_FROM_SENSOR.T, dtype=torch.float32, device=device).expand(
            len(rows), -1, -1
        )
        transformed = geometry.rotate_wrist_orientation(wrist, camera_rotation, right=right)
        for (_, view), value in zip(rows, transformed, strict=False):
            original = view["parameters"]["wrist_orient"]
            view["parameters"]["wrist_orient"] = value.to(device=original.device, dtype=original.dtype)
    output["camera_adapter"] = {
        "orientation": "explicit-Aria-Gen1-CW90-then-restore",
        "output_raster": "source",
        "output_camera_axes": "original_sensor",
    }
    return output
