# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Cartesian-friendly waypoint execution for serial arms (e.g. SO-100/101).

Joint-space linear interpolation between two IK solutions often makes the end-effector
cut through the workspace (e.g. driving horizontally into an object). Interpolating
SE(3) poses and re-solving IK each micro-step yields paths closer to a top-down arc
when waypoints share orientation and only move in Z (or a planned XY-then-Z path).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotionExecutionConfig:
    use_cartesian_interp: bool = True
    """If True, interpolate homogeneous poses and IK each step; else legacy joint lerp."""
    cartesian_step_m: float = 0.005
    """Target max translation per micro-step (meters)."""
    min_steps_per_segment: int = 8
    inter_step_sleep_s: float = 0.032
    """Sleep after each non-settling command (seconds)."""
    settle_last_step: bool = True
    """If True, only call wait_for_convergence on the last micro-step of each waypoint."""
    settle_timeout_s: float = 3.5
    settle_threshold_deg: float = 2.0
    settle_dt: float = 0.03
    # Legacy joint-only fallback
    max_joint_step_deg: float = 5.0


def wait_for_convergence(
    robot,
    target_joints: dict[str, float],
    *,
    timeout_s: float = 4.0,
    threshold_deg: float = 1.5,
    dt: float = 0.03,
) -> bool:
    start = time.time()
    max_err = float("inf")
    while time.time() - start < timeout_s:
        robot.send_action(target_joints)
        time.sleep(dt)
        obs = robot.get_observation()
        errors = []
        for key, target in target_joints.items():
            current = obs.get(key)
            if current is not None:
                errors.append(abs(float(current) - float(target)))
        if errors:
            max_err = max(errors)
            if max_err < threshold_deg:
                return True
    return False


def interpolate_se3(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate two 4x4 poses: linear translation, SLERP rotation."""
    from scipy.spatial.transform import Rotation

    t = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    r0 = Rotation.from_matrix(T0[:3, :3])
    r1 = Rotation.from_matrix(T1[:3, :3])
    r = r0 * Rotation.from_rotvec(alpha * (r1 * r0.inv()).as_rotvec())
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = t
    return T


def _translation_norm(T0: np.ndarray, T1: np.ndarray) -> float:
    return float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))


def _build_action(
    joint_target: np.ndarray,
    motor_names: list[str],
    *,
    gripper_open: bool,
    gripper_width_pct: float,
) -> dict[str, float]:
    action: dict[str, float] = {}
    for i, m in enumerate(motor_names):
        action[f"{m}.pos"] = float(joint_target[i])
    g = 100.0 if gripper_open else float(gripper_width_pct)
    action["gripper.pos"] = g
    return action


def execute_waypoints(
    robot,
    kinematics,
    waypoints: list,
    motor_names: list[str],
    motion: MotionExecutionConfig | None = None,
) -> None:
    """Execute grasp / place waypoints using Cartesian micro-steps when enabled."""
    motion = motion or MotionExecutionConfig()
    obs = robot.get_observation()
    current_joints = np.array(
        [float(obs[f"{m}.pos"]) for m in motor_names],
        dtype=np.float64,
    )

    for wp in waypoints:
        current_joints = execute_waypoint(robot, kinematics, wp, motor_names, motion, current_joints=current_joints)


def execute_waypoint(
    robot,
    kinematics,
    waypoint,
    motor_names: list[str],
    motion: MotionExecutionConfig | None = None,
    *,
    current_joints: np.ndarray | None = None,
) -> np.ndarray:
    """Execute a *single* waypoint and return the resulting joint vector.

    This is useful for "micro-step" agentic control loops that want to interleave:
    observe → reason → execute-one-waypoint → observe → reason → ...
    """
    motion = motion or MotionExecutionConfig()
    if current_joints is None:
        obs = robot.get_observation()
        current_joints = np.array(
            [float(obs[f"{m}.pos"]) for m in motor_names],
            dtype=np.float64,
        )

    wp = waypoint
    logger.info("  -> %s", getattr(wp, "label", "waypoint"))
    T_target = np.asarray(wp.pose_4x4, dtype=np.float64)
    T_start = kinematics.forward_kinematics(current_joints)
    delta_j = np.zeros_like(current_joints)

    if motion.use_cartesian_interp:
        dist = _translation_norm(T_start, T_target)
        n = max(motion.min_steps_per_segment, int(np.ceil(dist / max(motion.cartesian_step_m, 1e-6))))
        n = max(1, n)
    else:
        joint_end = kinematics.inverse_kinematics(current_joints, T_target)
        delta_j = joint_end - current_joints
        max_delta = float(np.max(np.abs(delta_j))) if delta_j.size > 0 else 0.0
        n = max(1, int(np.ceil(max_delta / max(motion.max_joint_step_deg, 1e-6))))

    for step in range(1, n + 1):
        alpha = step / n
        if motion.use_cartesian_interp:
            T_i = interpolate_se3(T_start, T_target, alpha)
            joint_target = kinematics.inverse_kinematics(current_joints, T_i)
        else:
            joint_target = current_joints + delta_j * alpha

        is_last = step == n
        label = getattr(wp, "label", "")
        if is_last:
            gripper_open = bool(wp.gripper_open)
            gripper_pct = float(wp.gripper_width_pct)
        elif label == "grasp" and not wp.gripper_open:
            # Stay open while moving into the grasp pose; close only on the last micro-step.
            gripper_open = True
            gripper_pct = float(np.clip(float(wp.gripper_width_pct) + 25.0, 45.0, 82.0))
        elif label == "place" and wp.gripper_open:
            # Keep holding until the final step of the place segment.
            gripper_open = False
            gripper_pct = float(wp.gripper_width_pct)
        else:
            gripper_open = bool(wp.gripper_open)
            gripper_pct = float(wp.gripper_width_pct)

        action = _build_action(
            joint_target[: len(motor_names)],
            motor_names,
            gripper_open=gripper_open,
            gripper_width_pct=gripper_pct,
        )

        if motion.settle_last_step and is_last:
            ok = wait_for_convergence(
                robot,
                action,
                timeout_s=motion.settle_timeout_s,
                threshold_deg=motion.settle_threshold_deg,
                dt=motion.settle_dt,
            )
            if not ok:
                logger.warning("  Segment end did not fully converge; continuing.")
        else:
            robot.send_action(action)
            time.sleep(motion.inter_step_sleep_s)

        current_joints = np.array(
            [action[f"{m}.pos"] for m in motor_names],
            dtype=np.float64,
        )

    if getattr(wp, "label", "") == "grasp" and not wp.gripper_open:
        time.sleep(0.45)
    elif getattr(wp, "label", "") == "place" and wp.gripper_open:
        time.sleep(0.28)

    return current_joints


def execute_waypoint_microsteps(
    robot,
    kinematics,
    waypoint,
    motor_names: list[str],
    motion: MotionExecutionConfig | None = None,
    *,
    current_joints: np.ndarray | None = None,
    microstep_index: int = 0,
    max_microsteps: int = 10,
) -> tuple[np.ndarray, int, bool]:
    """Execute up to `max_microsteps` micro-steps toward a single waypoint.

    Returns: (new_joints, new_microstep_index, done)
    where `microstep_index` counts how many micro-steps have been executed so far for this waypoint.
    """
    motion = motion or MotionExecutionConfig()
    if current_joints is None:
        obs = robot.get_observation()
        current_joints = np.array([float(obs[f"{m}.pos"]) for m in motor_names], dtype=np.float64)

    wp = waypoint
    T_target = np.asarray(wp.pose_4x4, dtype=np.float64)
    T_start = kinematics.forward_kinematics(current_joints)

    if motion.use_cartesian_interp:
        dist = _translation_norm(T_start, T_target)
        n = max(motion.min_steps_per_segment, int(np.ceil(dist / max(motion.cartesian_step_m, 1e-6))))
        n = max(1, n)
    else:
        joint_end = kinematics.inverse_kinematics(current_joints, T_target)
        delta_j = joint_end - current_joints
        max_delta = float(np.max(np.abs(delta_j))) if delta_j.size > 0 else 0.0
        n = max(1, int(np.ceil(max_delta / max(motion.max_joint_step_deg, 1e-6))))

    start_step = int(np.clip(microstep_index + 1, 1, n))
    end_step = int(np.clip(start_step + max(1, int(max_microsteps)) - 1, 1, n))

    for step in range(start_step, end_step + 1):
        alpha = step / n
        if motion.use_cartesian_interp:
            T_i = interpolate_se3(T_start, T_target, alpha)
            joint_target = kinematics.inverse_kinematics(current_joints, T_i)
        else:
            joint_target = current_joints + delta_j * alpha  # type: ignore[name-defined]

        is_last = step == n
        label = getattr(wp, "label", "")
        if is_last:
            gripper_open = bool(wp.gripper_open)
            gripper_pct = float(wp.gripper_width_pct)
        elif label == "grasp" and not wp.gripper_open:
            gripper_open = True
            gripper_pct = float(np.clip(float(wp.gripper_width_pct) + 25.0, 45.0, 82.0))
        elif label == "place" and wp.gripper_open:
            gripper_open = False
            gripper_pct = float(wp.gripper_width_pct)
        else:
            gripper_open = bool(wp.gripper_open)
            gripper_pct = float(wp.gripper_width_pct)

        action = _build_action(
            joint_target[: len(motor_names)],
            motor_names,
            gripper_open=gripper_open,
            gripper_width_pct=gripper_pct,
        )

        # Never do a long settle loop unless we truly reached the end of the waypoint.
        if motion.settle_last_step and is_last:
            ok = wait_for_convergence(
                robot,
                action,
                timeout_s=motion.settle_timeout_s,
                threshold_deg=motion.settle_threshold_deg,
                dt=motion.settle_dt,
            )
            if not ok:
                logger.warning("  Segment end did not fully converge; continuing.")
        else:
            robot.send_action(action)
            time.sleep(motion.inter_step_sleep_s)

        current_joints = np.array([action[f"{m}.pos"] for m in motor_names], dtype=np.float64)
        microstep_index = step

    done = microstep_index >= n
    if done:
        if getattr(wp, "label", "") == "grasp" and not wp.gripper_open:
            time.sleep(0.45)
        elif getattr(wp, "label", "") == "place" and wp.gripper_open:
            time.sleep(0.28)

    return current_joints, microstep_index, done
