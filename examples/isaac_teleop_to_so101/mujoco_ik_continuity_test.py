#!/usr/bin/env python

"""Deterministic engage/static/small-motion joint-continuity regression."""

from __future__ import annotations

import numpy as np

from .mujoco_sink import MOTOR_NAMES
from .mujoco_xr_control import XRToSO101Controller

POSES_DEG = (
    (15.0, -60.0, 70.0, 25.0, 30.0, 40.0),
    (-35.0, 45.0, -55.0, -30.0, -80.0, 65.0),
    (70.0, -20.0, 10.0, 60.0, 120.0, 10.0),
    # Valid calibrated pose used by the example, although shoulder_lift lies 3 degrees
    # outside the nominal URDF limit and therefore triggers Placo joint-space projection.
    (-4.0, -103.0, 97.0, 78.0, -65.0, 25.0),
)
SMALL_DELTAS_M = (
    np.array([0.005, 0.0, 0.0]),
    np.array([0.0, 0.005, 0.0]),
    np.array([0.0, 0.0, 0.005]),
)


def observation(q: np.ndarray) -> dict[str, float]:
    return {f"{name}.pos": float(value) for name, value in zip(MOTOR_NAMES, q, strict=True)}


def xr_action(position: np.ndarray, trigger: float = 0.3) -> dict:
    return {
        "grip_pos": position,
        "grip_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "squeeze": 1.0,
        "trigger": trigger,
    }


def arm_vector(action: dict[str, float]) -> np.ndarray:
    return np.array([float(action[f"{name}.pos"]) for name in MOTOR_NAMES[:5]])


def main() -> None:
    grip = np.array([0.20, -0.10, 0.25])
    max_engage = 0.0
    max_static = 0.0
    max_small_motion = 0.0

    for pose_values in POSES_DEG:
        q = np.array(pose_values)
        obs = observation(q)
        for delta in SMALL_DELTAS_M:
            controller = XRToSO101Controller(obs)
            engage = controller.compute(xr_action(grip), obs)
            static = controller.compute(xr_action(grip), obs)
            moved = controller.compute(xr_action(grip + delta), obs)
            if engage is None or static is None or moved is None:
                raise SystemExit("FAIL: engaged controller returned no action")

            engage_jump = float(np.max(np.abs(arm_vector(engage) - q[:5])))
            static_jump = float(np.max(np.abs(arm_vector(static) - q[:5])))
            small_motion_jump = float(np.max(np.abs(arm_vector(moved) - arm_vector(static))))
            max_engage = max(max_engage, engage_jump)
            max_static = max(max_static, static_jump)
            max_small_motion = max(max_small_motion, small_motion_jump)

            if not np.isclose(float(engage["gripper.pos"]), 70.0):
                raise SystemExit("FAIL: engage changed absolute gripper semantics")

    print(f"PASS poses={len(POSES_DEG)} directions={len(SMALL_DELTAS_M)}")
    print(
        f"max_engage_jump={max_engage:.6f}deg max_static_jump={max_static:.6f}deg "
        f"max_5mm_joint_step={max_small_motion:.3f}deg"
    )
    if max_engage > 1e-9:
        raise SystemExit("FAIL: engage frame did not strictly hold measured joints")
    if max_static > 1e-3:
        raise SystemExit("FAIL: stationary post-engage frame changed joints")
    if max_small_motion > 5.0:
        raise SystemExit("FAIL: first 5 mm motion selected a discontinuous IK solution")


if __name__ == "__main__":
    main()
