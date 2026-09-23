# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Ground arm-motion labels in measured joints using a calibrated ReBot URDF.

Called by feature extraction after selecting a short motion-consistent interval.
No robot control or IK is performed. Camera points still require visual tracking
or independently validated camera calibration; FK poses are not pixel coordinates.
"""

import hashlib
from pathlib import Path

import numpy as np


def extract_motion(states, state_names: list[str], config: dict, *, kinematics=None) -> list[dict]:
    """Return measured Cartesian direction labels in the explicitly named base frame."""
    if config["arm"] not in {"left", "right"} or config["units"] not in {"degrees", "radians"}:
        raise ValueError("Specify the arm and measured revolute-joint units")
    if not config.get("calibration") or not config.get("frame"):
        raise ValueError("FK requires calibration provenance and a named coordinate frame")
    urdf = Path(config["urdf"])
    joint_names, state_keys = config["joint_names"], config["state_keys"]
    if len(joint_names) != len(state_keys) or len(set(joint_names)) != len(joint_names):
        raise ValueError("Each modeled revolute joint needs exactly one state key")
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or len(states) < 2 or not np.isfinite(states).all():
        raise ValueError("Need at least two finite measured states")
    joints = states[:, [state_names.index(key) for key in state_keys]]
    if config["units"] == "radians":
        joints = np.rad2deg(joints)
    signs = np.asarray(config["signs"], dtype=float)
    offsets = np.asarray(config["offset_degrees"], dtype=float)
    if (
        signs.shape != (len(joint_names),)
        or offsets.shape != signs.shape
        or not np.isin(signs, [-1, 1]).all()
        or not np.isfinite(offsets).all()
    ):
        raise ValueError("Joint calibration needs one sign and finite offset per joint")
    joints = joints * signs + offsets
    if kinematics is None:
        from lerobot.model.kinematics import RobotKinematics

        kinematics = RobotKinematics(str(urdf), config["tool_frame"], joint_names)
    positions = np.asarray([kinematics.forward_kinematics(q)[:3, 3] for q in joints])
    if not np.isfinite(positions).all():
        raise ValueError("FK produced invalid poses")
    delta = positions[-1] - positions[0]
    threshold = float(config["deadband_m"])
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Use a measured positive displacement deadband")
    result = []
    for axis, displacement in enumerate(delta):
        if abs(displacement) < threshold:
            continue
        # Reject net-direction labels for trajectories that substantially reverse direction.
        travel = np.abs(np.diff(positions[:, axis])).sum()
        if travel > 1.5 * abs(displacement):
            raise ValueError("Split this reversing trajectory before assigning an atomic motion")
        direction = config["axis_directions"][axis]["positive" if displacement > 0 else "negative"]
        result.append(
            {
                "text": f"move the {config['arm']} gripper {direction}",
                "evidence": {
                    "method": "measured-joint FK",
                    "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
                    "calibration": config["calibration"],
                    "frame": config["frame"],
                    "tool_frame": config["tool_frame"],
                    "displacement_m": delta.tolist(),
                },
            }
        )
    return result
