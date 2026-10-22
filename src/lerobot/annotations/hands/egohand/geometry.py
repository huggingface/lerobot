# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Native SOMA156 actions in segment-start FLU, with metric scale preserved."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

FloatArray = NDArray[np.float64]
FLU_FROM_OPTICAL = np.asarray([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def require_rotations(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-2:] != (3, 3) or not np.isfinite(value).all():
        raise ValueError("Expected finite rotation matrices")
    if not np.allclose(value.swapaxes(-1, -2) @ value, np.eye(3), atol=2e-4) or not np.allclose(
        np.linalg.det(value), 1.0, atol=2e-4
    ):
        raise ValueError("Invalid rotation matrices; refusing implicit orthogonalization")
    return value


def optical_wrist(wrist, orientation):
    """Conjugate physical SOMA wrist parameters out of the rig joint basis.

    Defines identity at the native MHR neutral wrist. This is camera-equivariant,
    but does not claim that MHR's neutral axes equal MANO's anatomical axes.
    """
    wrist = require_rotations(wrist)
    orientation = require_rotations(orientation)
    return orientation.swapaxes(-1, -2) @ wrist @ orientation


def pack_action(wrist_action, fingers):
    wrist_action = np.asarray(wrist_action, dtype=np.float32)
    fingers = require_rotations(fingers)
    n = len(wrist_action)
    if wrist_action.shape != (n, 12) or fingers.shape != (n, 2, 24, 3, 3):
        raise ValueError("Expected EE12 and two native 24-joint SOMA articulations")
    if not np.isfinite(wrist_action).all():
        raise ValueError("Non-finite EE action")
    articulation = Rotation.from_matrix(fingers.reshape(-1, 3, 3)).as_rotvec()
    return np.concatenate((wrist_action, articulation.reshape(n, 144)), axis=1).astype(np.float32)


def native_parameters(result, geometry, variant):
    """Preserve physical-side native parameters on the original frame grid."""
    if variant not in ("raw", "smoothed"):
        raise ValueError(variant)
    n = result["source_frame_count"]
    fields = {
        "wrist_rotation_camera": np.full((n, 2, 3, 3), np.nan),
        "soma_wrist_orient": np.full((n, 2, 1, 3, 3), np.nan),
        "soma_hand_pose": np.full((n, 2, 24, 3, 3), np.nan),
        "soma_identity": np.full((n, 2, 5), np.nan),
        "soma_scales": np.full((n, 2, 27), np.nan),
    }
    orientations = []
    for side in ("left", "right"):
        layer = geometry._layer(side, geometry._device_anchor.device)
        bind = layer.bind_pose_world[0].detach().cpu().double().numpy()
        rest = layer.t_pose_world[0].detach().cpu().double().numpy()
        orientations.append(np.linalg.solve(bind, rest)[:3, :3])
    seen = set()
    for frame in result["frames"]:
        i = frame["frame_id"]
        if i not in range(n):
            raise ValueError("Invalid frame index")
        for row in frame["instances"]:
            h = ("left", "right").index(row["side"])
            if (i, h) in seen:
                raise ValueError("Duplicate physical-side prediction")
            seen.add((i, h))
            view = row if variant == "raw" else row["smoothed"]
            params = view["parameters"]
            for source, target in (
                ("wrist_orient", "soma_wrist_orient"),
                ("hand_pose", "soma_hand_pose"),
                ("identity_coeffs", "soma_identity"),
                ("scale_params", "soma_scales"),
            ):
                value = np.asarray(params[source])
                if value.shape != fields[target][i, h].shape:
                    raise ValueError(f"Wrong native parameter shape: {source}")
                fields[target][i, h] = value
            fields["wrist_rotation_camera"][i, h] = optical_wrist(params["wrist_orient"][0], orientations[h])
    fields["soma_rig_orientation"] = np.stack(orientations)
    return fields


def fuse_segment_to_flu(
    joints_camera: FloatArray,
    wrist_rotation_camera: FloatArray,
    world_from_camera: FloatArray,
    *,
    scale: float = 1.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return 12D wrist actions and 21-joint geometry in segment-start FLU."""
    joints = np.asarray(joints_camera, dtype=np.float64)
    wrist_rot = np.asarray(wrist_rotation_camera, dtype=np.float64)
    poses = np.asarray(world_from_camera, dtype=np.float64)
    n = len(joints)
    if joints.shape != (n, 2, 21, 3):
        raise ValueError("joints_camera must have shape (N, 2, 21, 3)")
    if wrist_rot.shape != (n, 2, 3, 3):
        raise ValueError("wrist rotations must have shape (N, 2, 3, 3)")
    if poses.shape != (n, 4, 4):
        raise ValueError("camera transforms must have shape (N, 4, 4)")
    if n == 0:
        return np.empty((0, 12), dtype=np.float32), np.empty((0, 2, 21, 3), dtype=np.float32)

    world_joints = np.empty_like(joints)
    world_wrist_rot = np.empty_like(wrist_rot)
    for frame in range(n):
        rotation = poses[frame, :3, :3]
        translation = poses[frame, :3, 3]
        world_joints[frame] = np.einsum("ij,hkj->hki", rotation, joints[frame] * scale) + translation
        world_wrist_rot[frame] = np.einsum("ij,hjk->hik", rotation, wrist_rot[frame])

    first_from_world = np.linalg.inv(poses[0])
    first_rotation = first_from_world[:3, :3]
    first_translation = first_from_world[:3, 3]
    relative_joints_optical = np.einsum("ij,thkj->thki", first_rotation, world_joints) + first_translation
    relative_rot_optical = np.einsum("ij,tnjk->tnik", first_rotation, world_wrist_rot)
    joints_flu = np.einsum("ij,thkj->thki", FLU_FROM_OPTICAL, relative_joints_optical)
    wrist_rot_flu = np.einsum("ij,thjk,lk->thil", FLU_FROM_OPTICAL, relative_rot_optical, FLU_FROM_OPTICAL)

    action = np.empty((n, 12), dtype=np.float32)
    for hand in range(2):
        offset = hand * 6
        action[:, offset : offset + 3] = joints_flu[:, hand, 0].astype(np.float32)
        action[:, offset + 3 : offset + 6] = (
            Rotation.from_matrix(wrist_rot_flu[:, hand]).as_rotvec().astype(np.float32)
        )
    return action, joints_flu.astype(np.float32)
