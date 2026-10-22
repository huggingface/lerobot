#!/usr/bin/env python3
"""
Auto-reset and cube-grab utility for the OMX robot arm.

Provides:
  - grab_cube(robot): sweep workspace, center cube, close gripper
  - place_cube(robot): carry cube to a random position, release

Standalone usage (run from repo root):
    python -m examples.omx.reset_environment --port /dev/ttyACM1 --mode grab
    python -m examples.omx.reset_environment --port /dev/ttyACM1 --mode grab_and_place

Joint range: -100 to 100 for arm joints; gripper: 50 = closed, 80 = open.

To read current joint values for calibration, add after robot.connect():
    obs = robot.get_observation()
    print({k: round(obs[k], 1) for k in JOINT_NAMES})
    robot.disconnect(); raise SystemExit

Parallel-to-ground IK: wrist_flex = WRIST_HORIZONTAL_OFFSET - shoulder_lift - elbow_flex.
Linear interpolation preserves this constraint between any two poses that satisfy it.
"""

import argparse
import logging

import numpy as np

from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
from lerobot.robots.robot import Robot
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

# ── Poses ─────────────────────────────────────────────────────────────────────

HOME_POSE = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -50.0,
    "elbow_flex.pos": 50.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 60.0,
}

SWEEP_WAYPOINTS = [
    {
        "shoulder_pan.pos": -60.0,
        "shoulder_lift.pos": 50.0,
        "elbow_flex.pos": -60.0,
        "wrist_flex.pos": -20.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 60.0,
    },
    {
        "shoulder_pan.pos": -30.0,
        "shoulder_lift.pos": 50.0,
        "elbow_flex.pos": -60.0,
        "wrist_flex.pos": -5.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 60.0,
    },
    {
        "shoulder_pan.pos": 20.0,
        "shoulder_lift.pos": 50.0,
        "elbow_flex.pos": -55.0,
        "wrist_flex.pos": -5.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 60.0,
    },
]

# ── Motion parameters ─────────────────────────────────────────────────────────

CONTROL_HZ = 30
APPROACH_SPEED = 50.0
SWEEP_SPEED = 40.0

# ── Grab-sequence parameters ──────────────────────────────────────────────────

GRAB_PAN = 0.0
SWEEP_LEFT_PAN = -60.0
SWEEP_RIGHT_PAN = 60.0
SWEEP_END_OFFSET = 5.0  # stop before center so the cube isn't pushed past GRAB_PAN
SWEEP_END_PAN_RANGE = (15.0, 20.0)

SWEEP_LOW_SHOULDER_LIFT = 50.0
SWEEP_LOW_ELBOW_FLEX_START = -60.0
SWEEP_LOW_ELBOW_FLEX_END = -55.0

SWEEP_HIGH_WRIST_FLEX = -20.0  # wrist tilted up during high approach to clear obstacles

PUSH_START_SHOULDER_LIFT = 0.0
PUSH_START_ELBOW_FLEX = 45.0
PUSH_END_SHOULDER_LIFT = 50.0
PUSH_END_ELBOW_FLEX = -50.0
# Subtracted from shoulder_lift during the push sweep to clear the platform surface.
# Does not affect the grab-target interpolation in record_grab.py.
PUSH_RAISE_OFFSET = 5.0

WRIST_HORIZONTAL_OFFSET = 0.0  # tune if gripper tilts during push: + tilts nose up, - down
GRIPPER_CLOSE_POS = 50.0

PLACE_LEFT_PAN_RANGE = (5.0, 30.0)  # random pan range for cube placement on the left side
PLACE_REACH_RANGE = (0.1, 0.7)  # 0 = arm retracted (PUSH_START), 1 = fully extended (PUSH_END)

JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def pose_to_array(pose: dict) -> np.ndarray:
    return np.array([pose[k] for k in JOINT_NAMES])


def array_to_pose(arr: np.ndarray) -> dict:
    return {k: float(arr[i]) for i, k in enumerate(JOINT_NAMES)}


def horizontal_wrist_flex(shoulder_lift: float, elbow_flex: float) -> float:
    return WRIST_HORIZONTAL_OFFSET - shoulder_lift - elbow_flex


def _low_sweep_pose(pan: float, elbow_flex: float, wrist_flex: float | None = None) -> dict:
    sl = SWEEP_LOW_SHOULDER_LIFT
    return {
        "shoulder_pan.pos": pan,
        "shoulder_lift.pos": sl,
        "elbow_flex.pos": elbow_flex,
        "wrist_flex.pos": horizontal_wrist_flex(sl, elbow_flex) if wrist_flex is None else wrist_flex,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 60.0,
    }


def _high_sweep_pose(pan: float) -> dict:
    return {**HOME_POSE, "shoulder_pan.pos": pan, "wrist_flex.pos": SWEEP_HIGH_WRIST_FLEX}


def _push_pose(shoulder_lift: float, elbow_flex: float, pan: float = GRAB_PAN, gripper: float = 70.0) -> dict:
    return {
        "shoulder_pan.pos": pan,
        "shoulder_lift.pos": shoulder_lift,
        "elbow_flex.pos": elbow_flex,
        "wrist_flex.pos": horizontal_wrist_flex(shoulder_lift, elbow_flex),
        "wrist_roll.pos": 0.0,
        "gripper.pos": gripper,
    }


def move_to_pose(robot: Robot, target: dict, speed: float) -> None:
    """Interpolate from current position to target at the given speed (units/s)."""
    obs = robot.get_observation()
    current = np.array([obs[k] for k in JOINT_NAMES])
    goal = pose_to_array(target)

    max_distance = float(np.max(np.abs(goal - current)))
    if max_distance < 0.5:
        return

    n_steps = max(1, int(max_distance / speed * CONTROL_HZ))
    dt = 1.0 / CONTROL_HZ
    for step in range(1, n_steps + 1):
        t = step / n_steps
        robot.send_action(array_to_pose(current + t * (goal - current)))
        precise_sleep(dt)


# ── Sequences ─────────────────────────────────────────────────────────────────


def grab_cube(robot: Robot) -> None:
    """Left sweep → right sweep → extend arm parallel to ground → close gripper."""
    move_to_pose(robot, HOME_POSE, APPROACH_SPEED)

    for pan, end_pan in [
        (SWEEP_LEFT_PAN, GRAB_PAN - SWEEP_END_OFFSET),
        (SWEEP_RIGHT_PAN, GRAB_PAN + SWEEP_END_OFFSET),
    ]:
        logger.info(f"Sweeping {'left' if pan < 0 else 'right'} → center...")
        move_to_pose(robot, _high_sweep_pose(pan), APPROACH_SPEED)
        move_to_pose(
            robot, _low_sweep_pose(pan, SWEEP_LOW_ELBOW_FLEX_START, wrist_flex=-20.0), APPROACH_SPEED
        )
        move_to_pose(robot, _low_sweep_pose(end_pan, SWEEP_LOW_ELBOW_FLEX_END, wrist_flex=0.0), SWEEP_SPEED)
        move_to_pose(robot, HOME_POSE, APPROACH_SPEED)

    logger.info("Extending to push cube into gripper...")
    move_to_pose(
        robot,
        _push_pose(PUSH_START_SHOULDER_LIFT - PUSH_RAISE_OFFSET, PUSH_START_ELBOW_FLEX),
        APPROACH_SPEED,
    )
    move_to_pose(
        robot,
        _push_pose(PUSH_END_SHOULDER_LIFT - PUSH_RAISE_OFFSET, PUSH_END_ELBOW_FLEX),
        SWEEP_SPEED,
    )

    logger.info("Closing gripper...")
    move_to_pose(
        robot,
        _push_pose(PUSH_END_SHOULDER_LIFT, PUSH_END_ELBOW_FLEX, gripper=GRIPPER_CLOSE_POS),
        APPROACH_SPEED,
    )

    logger.info("Grab complete.")


def place_cube(robot: Robot) -> tuple[float, float]:
    """Carry the cube (gripper closed) to a random position on the left side, then release.

    Returns:
        (pan, t): pan angle and reach scalar [0, 1] of the placement position.
    """
    pan = float(np.random.uniform(*PLACE_LEFT_PAN_RANGE))
    t = float(np.random.uniform(*PLACE_REACH_RANGE))
    sl = PUSH_START_SHOULDER_LIFT + t * (PUSH_END_SHOULDER_LIFT - PUSH_START_SHOULDER_LIFT)
    ef = PUSH_START_ELBOW_FLEX + t * (PUSH_END_ELBOW_FLEX - PUSH_START_ELBOW_FLEX)
    logger.info(f"Placing cube at pan={pan:.1f}, reach={t:.2f}...")

    move_to_pose(robot, {**HOME_POSE, "gripper.pos": GRIPPER_CLOSE_POS}, APPROACH_SPEED)
    move_to_pose(
        robot, {**HOME_POSE, "shoulder_pan.pos": pan, "gripper.pos": GRIPPER_CLOSE_POS}, APPROACH_SPEED
    )
    move_to_pose(robot, _push_pose(sl, ef, pan=pan, gripper=GRIPPER_CLOSE_POS), APPROACH_SPEED)
    move_to_pose(robot, _push_pose(sl, ef, pan=pan, gripper=80.0), APPROACH_SPEED)
    move_to_pose(robot, HOME_POSE, APPROACH_SPEED)
    logger.info("Place complete.")
    return pan, t


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="OMX arm reset / grab script")
    parser.add_argument("--port", default="/dev/ttyACM1")
    parser.add_argument("--robot_id", default="omx_follower")
    parser.add_argument("--mode", choices=["grab", "grab_and_place"], default="grab_and_place")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    robot = OmxFollower(OmxFollowerConfig(port=args.port, id=args.robot_id))
    robot.connect(calibrate=True)

    try:
        if args.mode == "grab":
            grab_cube(robot)
        elif args.mode == "grab_and_place":
            grab_cube(robot)
            place_cube(robot)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
