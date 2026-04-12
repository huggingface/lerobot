"""
Bringup script for the SO-101 follower arm.

Moves the robot through a sequence of pre-defined configurations to verify
that all joints are responsive and calibrated. Interpolates smoothly between
poses for safety.

Example:

```shell
python -m lerobot.scripts.lerobot_bringup \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5AAF2879361 \
    --robot.id=follower
```
"""

import logging
import time
from dataclasses import dataclass
from pprint import pformat

import rerun as rr

from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so101_follower,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)

# Joint names in order
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# ---------------------------------------------------------------------------
# Pre-defined configurations (normalized values)
#   Arm joints: [-100, 100]   Gripper: [0, 100]
# ---------------------------------------------------------------------------
HOME_CONFIG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 50.0,
}

STOW_CONFIG = {
    "shoulder_pan": -24.0,
    "shoulder_lift": 100.0,
    "elbow_flex": -38.0,
    "wrist_flex": 29.0,
    "wrist_roll": 0.0,
    "gripper": 50.0,
}

CONFIGURATIONS: list[tuple[str, dict[str, float]]] = [
    ("Home (centered)", HOME_CONFIG),
    (
        "Shoulder pan left",
        {
            "shoulder_pan": 15.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
        },
    ),
    (
        "Shoulder pan center",
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
        },
    ),
    (
        "Shoulder pan right",
        {
            "shoulder_pan": -40.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
        },
    ),
    (
        "Shoulder lift up",
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": -30.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
        },
    ),
    (
        "Elbow flex",
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 40.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
        },
    ),
    (
        "Wrist flex",
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 40.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
        },
    ),
    (
        "Wrist roll",
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 50.0,
            "gripper": 50.0,
        },
    ),
    (
        "Gripper open",
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 95.0,
        },
    ),
    (
        "Gripper half",
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
        },
    ),
    (
        "Gripper close",
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 5.0,
        },
    ),
    (
        "Ready pose (arm raised, elbow bent)",
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": -20.0,
            "elbow_flex": 30.0,
            "wrist_flex": -10.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
        },
    ),
    (
        "Folding 1/3",
        {
            "shoulder_pan": -8.0,
            "shoulder_lift": 20.0,
            "elbow_flex": 7.0,
            "wrist_flex": 3.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
        },
    ),
    (
        "Folding 2/3",
        {
            "shoulder_pan": -16.0,
            "shoulder_lift": 60.0,
            "elbow_flex": -15.0,
            "wrist_flex": 16.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
        },
    ),
    ("Stow (folded, gravity-safe)", STOW_CONFIG),
]


@dataclass
class BringupConfig:
    robot: RobotConfig
    # Number of interpolation steps between configurations
    interpolation_steps: int = 50
    # Target loop rate in Hz for interpolation steps
    fps: int = 50
    # Seconds to hold each configuration before moving to the next
    hold_time_s: float = 1.0
    # If True, wait for user to press Enter between configurations
    interactive: bool = True
    # Display joint data in Rerun viewer
    display_data: bool = False


def interpolate(
    start: dict[str, float], end: dict[str, float], steps: int
) -> list[dict[str, float]]:
    """Generate linearly interpolated waypoints between start and end."""
    waypoints = []
    for i in range(1, steps + 1):
        alpha = i / steps
        waypoints.append({k: start[k] + alpha * (end[k] - start[k]) for k in start})
    return waypoints


def read_current_config(robot: Robot) -> dict[str, float]:
    """Read current joint positions and strip the '.pos' suffix."""
    obs = robot.get_observation()
    return {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}


def send_config(robot: Robot, config: dict[str, float]) -> None:
    """Send a joint configuration to the robot."""
    action = {f"{k}.pos": v for k, v in config.items()}
    robot.send_action(action)


def print_config(label: str, config: dict[str, float]) -> None:
    """Pretty-print a joint configuration."""
    max_len = max(len(k) for k in config)
    print(f"\n  {label}")
    print(f"  {'Joint':<{max_len}}   Value")
    print(f"  {'-' * max_len}   -----")
    for joint in JOINTS:
        if joint in config:
            print(f"  {joint:<{max_len}}   {config[joint]:>7.2f}")


def move_to_config(
    robot: Robot,
    target: dict[str, float],
    steps: int,
    fps: int,
    display_data: bool = False,
    config_name: str = "",
) -> None:
    """Smoothly interpolate from the current position to the target configuration."""
    current = read_current_config(robot)
    waypoints = interpolate(current, target, steps)
    for wp in waypoints:
        loop_start = time.perf_counter()
        send_config(robot, wp)

        if display_data:
            obs = robot.get_observation()
            action = {f"{k}.pos": v for k, v in wp.items()}
            log_rerun_data(observation=obs, action=action)
            if config_name:
                rr.log("bringup/config_name", rr.TextLog(config_name))

        dt = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt)


@parser.wrap()
def bringup(cfg: BringupConfig):
    logging.basicConfig(level=logging.INFO)
    logger.info(pformat(cfg))

    if cfg.display_data:
        init_rerun(session_name="bringup")

    print("=" * 60)
    print("  SO-101 Bringup Procedure")
    print("=" * 60)

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    print(f"\nRobot connected: {robot.is_connected}")

    # Show starting position
    current = read_current_config(robot)
    print_config("Current position", current)

    try:
        for i, (name, target) in enumerate(CONFIGURATIONS):
            step_label = f"[{i + 1}/{len(CONFIGURATIONS)}] {name}"
            print(f"\n{'=' * 60}")
            print(f"  {step_label}")
            print_config("Target", target)

            if cfg.interactive:
                input("\n  Press ENTER to move (Ctrl+C to abort)...")

            print("  Moving...")
            move_to_config(
                robot, target, cfg.interpolation_steps, cfg.fps,
                display_data=cfg.display_data, config_name=name,
            )

            # Read back and show actual position
            actual = read_current_config(robot)
            print_config("Actual position", actual)

            # Check deviation
            max_error = max(abs(actual[j] - target[j]) for j in target)
            if max_error > 5.0:
                print(f"\n  WARNING: max joint error = {max_error:.2f} (threshold: 5.0)")
            else:
                print(f"\n  OK (max error: {max_error:.2f})")

            if cfg.hold_time_s > 0:
                time.sleep(cfg.hold_time_s)

        print(f"\n{'=' * 60}")
        print("  Bringup complete. All configurations visited.")
        print("=" * 60)

        # Move to stow and hold with torque
        print("\n  Moving to Stow (gravity-safe) and holding position...")
        print("  Press Ctrl+C to release and disconnect.\n")
        move_to_config(
            robot, STOW_CONFIG, cfg.interpolation_steps, cfg.fps,
            display_data=cfg.display_data, config_name="Stow (holding)",
        )
        while True:
            loop_start = time.perf_counter()
            send_config(robot, STOW_CONFIG)
            if cfg.display_data:
                obs = robot.get_observation()
                action = {f"{k}.pos": v for k, v in STOW_CONFIG.items()}
                log_rerun_data(observation=obs, action=action)
            dt = time.perf_counter() - loop_start
            busy_wait(1 / cfg.fps - dt)

    except KeyboardInterrupt:
        print("\n\nStopping. Robot will hold briefly then disconnect.")
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        robot.disconnect()
        print("Robot disconnected.")


def main():
    bringup()


if __name__ == "__main__":
    main()
