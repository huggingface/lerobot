# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
import numpy as np
import torch

from lerobot.annotations.hands.egohand.orientation import (
    UPRIGHT_FROM_SENSOR,
    UprightFrames,
    restore_annotations,
    restore_predictions,
    sensor_pixels,
    upright_intrinsics,
)


def test_calibrated_ray_rotation_and_round_trip():
    k = np.array([[400.0, 0, 301.2], [0, 500.0, 220.3], [0, 0, 1]])
    points = np.random.default_rng(2).normal(size=(40, 3))
    points[:, 2] = np.abs(points[:, 2]) + 1
    upright = points @ UPRIGHT_FROM_SENSOR.T
    rotated_k = upright_intrinsics(k, 480)
    pixels = (points / points[:, 2:]) @ k.T
    rotated_pixels = (upright / upright[:, 2:]) @ rotated_k.T
    np.testing.assert_allclose(sensor_pixels(rotated_pixels[:, :2], 480), pixels[:, :2], atol=1e-12)
    np.testing.assert_allclose(upright @ UPRIGHT_FROM_SENSOR, points, atol=1e-12)


def test_exact_raster_rotation():
    source = np.arange(2 * 4 * 6 * 3, dtype=np.uint8).reshape(2, 4, 6, 3)
    rotated = UprightFrames(source)
    np.testing.assert_array_equal(np.rot90(rotated[0], k=1), source[0])
    assert len(rotated[:]) == 2
    annotations = {
        0: {
            "intrinsics": np.eye(3),
            "instances": [{"side": "left", "bbox": [1.0, 3.0, 2.0], "bbox_observed": True}],
        }
    }
    restored = restore_annotations(annotations, np.eye(3), 4)
    assert restored[0]["instances"][0]["bbox"] == [3.0, 2.0, 2.0]
    assert restored[0]["instances"][0]["side"] == "left"
    assert annotations[0]["instances"][0]["bbox"] == [1.0, 3.0, 2.0]


def test_geometry_restores_coordinates_without_side_flip():
    class Geometry:
        _device_anchor = torch.empty(0)

        def rotate_wrist_orientation(self, wrist, camera, *, right):
            assert right.tolist() == [True]
            return camera[:, None] @ wrist

    k = np.array([[10.0, 0, 3.0], [0, 10.0, 2.0], [0, 0, 1]])
    original = torch.tensor([[0.1, 0.2, 1.0]])
    rotated = original @ torch.tensor(UPRIGHT_FROM_SENSOR.T, dtype=torch.float32)
    pixels = rotated / rotated[:, 2:] @ torch.tensor(upright_intrinsics(k, 8).T, dtype=torch.float32)
    row = {
        "side": "right",
        "bbox": torch.tensor([5.0, 4.0, 3.0]),
        "intrinsics": torch.eye(3),
        "parameters": {"wrist_orient": torch.eye(3)[None]},
        "joints_camera": rotated,
        "vertices_camera": rotated.clone(),
        "camera_translation": rotated[0].clone(),
        "keypoints_2d": pixels[:, :2],
        "keypoints_crop": torch.zeros(1, 2),
        "bbox_observed": False,
    }
    result = {"frames": [{"frame_id": 0, "instances": [row]}]}
    restored = restore_predictions(result, Geometry(), k, 10, 8)
    target = restored["frames"][0]["instances"][0]
    torch.testing.assert_close(target["joints_camera"], original)
    assert target["side"] == "right" and not target["bbox_observed"]
    torch.testing.assert_close(
        target["keypoints_2d"], (original / original[:, 2:] @ torch.tensor(k.T, dtype=torch.float32))[:, :2]
    )
    torch.testing.assert_close(row["joints_camera"], rotated)
