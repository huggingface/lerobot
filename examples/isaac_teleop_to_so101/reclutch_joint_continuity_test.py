#!/usr/bin/env python

"""Deterministic SO-101 XR re-clutch continuity regression without hardware or MuJoCo."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.types import RobotAction, RobotObservation

from .common import (
    IK_ORIENTATION_WEIGHT,
    MAX_EE_STEP_M,
    IKJointRebase,
    build_xr_kinematics,
    initialize_ik_engage,
)
from .isaac_teleop import Clutch, MapXRControllerActionToRobotAction

MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
POSES_DEG = (
    (15.0, -60.0, 70.0, 25.0, 30.0, 40.0),
    (-35.0, 45.0, -55.0, -30.0, -80.0, 65.0),
    (70.0, -20.0, 10.0, 60.0, 120.0, 10.0),
    # Calibrated reset pose used by the example. shoulder_lift is intentionally 3 degrees
    # outside the nominal URDF limit, exercising Placo's feasible-space projection.
    (-4.0, -103.0, 97.0, 78.0, -65.0, 25.0),
)
SMALL_DELTAS_M = (
    np.array([0.005, 0.0, 0.0]),
    np.array([0.0, 0.005, 0.0]),
    np.array([0.0, 0.0, 0.005]),
)
GRIP_POS = np.array([0.20, -0.10, 0.25])
GRIP_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def observation(q: np.ndarray) -> RobotObservation:
    return {f"{name}.pos": float(value) for name, value in zip(MOTOR_NAMES, q, strict=True)}


def build_processor(kinematics):
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapXRControllerActionToRobotAction(),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=MAX_EE_STEP_M,
                raise_on_jump=False,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=list(MOTOR_NAMES),
                initial_guess_current_joints=False,
                orientation_weight=IK_ORIENTATION_WEIGHT,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def ee_action(position: np.ndarray, quaternion: np.ndarray) -> RobotAction:
    return {
        "ee_pose": np.concatenate([position, quaternion]).astype(np.float32),
        "closedness": 0.3,
    }


def arm_vector(action: RobotAction) -> np.ndarray:
    return np.array([float(action[f"{name}.pos"]) for name in MOTOR_NAMES[:5]])


def main() -> None:
    max_position_reanchor_m = 0.0
    max_rotation_reanchor = 0.0
    max_engage_jump_deg = 0.0
    max_static_jump_deg = 0.0
    max_small_motion_deg = 0.0

    for pose_index, pose_values in enumerate(POSES_DEG):
        q = np.array(pose_values)
        obs = observation(q)
        stale_q = np.array(POSES_DEG[(pose_index + 1) % len(POSES_DEG)])

        for delta in SMALL_DELTAS_M:
            kinematics = build_xr_kinematics(list(MOTOR_NAMES))
            stale_pose = kinematics.forward_kinematics(stale_q)
            measured_pose = kinematics.forward_kinematics(q)
            processor = build_processor(kinematics)

            # Start from a deliberately stale command pose. engage() must replace both its
            # position and orientation with the complete measured FK pose.
            clutch = Clutch(stale_pose)
            clutch.engage(GRIP_POS, GRIP_QUAT, measured_base_T_ee=measured_pose)
            processor.reset()
            base_pos, base_quat = clutch.rebase(GRIP_POS, GRIP_QUAT)
            position_reanchor = float(np.linalg.norm(base_pos - measured_pose[:3, 3]))
            rotation_reanchor = float(
                np.max(np.abs(Rotation.from_quat(base_quat).as_matrix() - measured_pose[:3, :3]))
            )

            base_action = ee_action(base_pos, base_quat)
            ik_engage = initialize_ik_engage(processor, base_action, obs, list(MOTOR_NAMES))
            joint_rebase = IKJointRebase(list(MOTOR_NAMES))
            engage = joint_rebase.engage(ik_engage, obs)
            static = joint_rebase.apply(processor((ee_action(base_pos, base_quat), obs)))

            moved_pos, moved_quat = clutch.rebase(GRIP_POS + delta, GRIP_QUAT)
            moved = joint_rebase.apply(processor((ee_action(moved_pos, moved_quat), obs)))

            engage_jump = float(np.max(np.abs(arm_vector(engage) - q[:5])))
            static_jump = float(np.max(np.abs(arm_vector(static) - q[:5])))
            small_motion = float(np.max(np.abs(arm_vector(moved) - arm_vector(static))))

            max_position_reanchor_m = max(max_position_reanchor_m, position_reanchor)
            max_rotation_reanchor = max(max_rotation_reanchor, rotation_reanchor)
            max_engage_jump_deg = max(max_engage_jump_deg, engage_jump)
            max_static_jump_deg = max(max_static_jump_deg, static_jump)
            max_small_motion_deg = max(max_small_motion_deg, small_motion)

            if not np.isclose(float(engage["gripper.pos"]), 70.0):
                raise SystemExit("FAIL: engage changed absolute gripper semantics")

    print(f"PASS poses={len(POSES_DEG)} directions={len(SMALL_DELTAS_M)}")
    print(f"reanchor_position={max_position_reanchor_m:.9f}m reanchor_rotation={max_rotation_reanchor:.9f}")
    print(
        f"engage_jump={max_engage_jump_deg:.6f}deg "
        f"static_jump={max_static_jump_deg:.6f}deg "
        f"max_5mm_joint_step={max_small_motion_deg:.3f}deg"
    )
    if max_position_reanchor_m > 1e-9 or max_rotation_reanchor > 1e-9:
        raise SystemExit("FAIL: clutch did not re-anchor the complete measured FK pose")
    if max_engage_jump_deg > 1e-9:
        raise SystemExit("FAIL: engage frame did not strictly hold measured joints")
    if max_static_jump_deg > 1e-3:
        raise SystemExit("FAIL: stationary post-engage frame changed joints")
    if max_small_motion_deg > 5.0:
        raise SystemExit("FAIL: first 5 mm motion selected a discontinuous IK solution")


if __name__ == "__main__":
    main()
