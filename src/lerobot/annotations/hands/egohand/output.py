# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Export only directly observed native bimanual spans, with no model substitution."""

from pathlib import Path

import numpy as np

from ..config import ACTION_SCHEMA, RECIPE

NATIVE_FIELDS = {
    "soma_wrist_orient": (2, 1, 3, 3),
    "soma_hand_pose": (2, 24, 3, 3),
    "soma_identity": (2, 5),
    "soma_scales": (2, 27),
    "wrist_rotation_camera": (2, 3, 3),
}


def observed_spans(direct, camera_valid, *, minimum_frames=30):
    direct = np.asarray(direct)
    if direct.dtype != np.bool_ or direct.ndim != 2 or direct.shape[1] != 2:
        raise ValueError("Direct hand observations must be a boolean (N, 2) array")
    if np.shape(camera_valid) != (len(direct),):
        raise ValueError("Camera validity must match the source frame grid")
    adjacent = np.zeros_like(direct)
    adjacent[1:] |= direct[:-1]
    adjacent[:-1] |= direct[1:]
    supported = direct & adjacent
    accepted = supported.all(axis=1) & camera_valid
    edges = np.flatnonzero(np.diff(np.pad(accepted.astype(np.int8), (1, 1))))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2) if b - a >= minimum_frames]


def export_actions(prediction: Path, camera: Path, directory: Path, *, frames: int, fps: float):
    from .geometry import fuse_segment_to_flu, pack_action, require_rotations

    with np.load(prediction, allow_pickle=False) as archive:
        values = dict(archive)
    with np.load(camera, allow_pickle=False) as archive:
        poses = archive["camera_pose"]
        source_index = archive["source_frame_index"]
        timestamps = archive["timestamp_s"]
    if poses.shape != (frames, 4, 4) or not np.array_equal(source_index, np.arange(frames)):
        raise ValueError("Camera trajectory does not cover the complete source frame grid")
    if timestamps.shape != (frames,) or not np.allclose(timestamps, source_index / fps, rtol=0, atol=1e-6):
        raise ValueError("Camera timestamps differ from the source video")
    direct = values["direct_valid"]
    joints = values["joints_camera"]
    if direct.shape != (frames, 2) or joints.shape != (frames, 2, 21, 3):
        raise ValueError("Native hand predictions differ from the source frame grid")
    finite = np.isfinite(joints).all(axis=(2, 3))
    for name, shape in NATIVE_FIELDS.items():
        if values[name].shape != (frames, *shape):
            raise ValueError(f"Invalid native SOMA parameter shape: {name}")
        finite &= np.isfinite(values[name]).reshape(frames, 2, -1).all(axis=-1)
    spans = observed_spans(direct & finite, np.isfinite(poses).all(axis=(1, 2)))
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for start, end in spans:
        selected = slice(start, end)
        wrists, world = fuse_segment_to_flu(
            joints[selected],
            require_rotations(values["wrist_rotation_camera"][selected]),
            poses[selected],
            scale=1.0,
        )
        path = directory / f"soma-{start:09d}-{end:09d}.npz"
        np.savez_compressed(
            path,
            action=pack_action(wrists, values["soma_hand_pose"][selected]),
            source_frame_index=source_index[selected],
            timestamp_s=timestamps[selected],
            hand_joints_camera=joints[selected],
            hand_joints_world=world,
            camera_world_from_optical=poses[selected],
            **{name: values[name][selected] for name in NATIVE_FIELDS},
        )
        paths.append(str(path))
    return {
        "recipe": RECIPE,
        "action_schema": ACTION_SCHEMA,
        "rectified_frames": frames,
        "rectified_fps": fps,
        "accepted_frames": sum(end - start for start, end in spans),
        "segments": paths,
        "motion_infilling": False,
        "gt_scale_alignment": False,
        "camera": "DROID-W+MoGe Small",
    }
