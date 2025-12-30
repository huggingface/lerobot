#!/usr/bin/env python

"""
Mirrored bimanual teleoperation:
- Right leader → Left follower
- Left leader → Right follower
- Invert twist joints: shoulder_pan (base) and wrist_roll (second-last before gripper)

Example:

```shell
python -m lerobot.scripts.lerobot_teleoperate_mirrored \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --fps=60 \
  --display_data=true
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.configs import parser
from lerobot.robots import (  # ensure types are registered
    Robot,
    RobotConfig,
    bi_so100_follower,  # noqa: F401
    make_robot_from_config,
)
from lerobot.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,  # noqa: F401
    make_teleoperator_from_config,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Twist joints to invert sign on both arms
TWIST_JOINT_NAMES = ("shoulder_pan.pos", "wrist_roll.pos")


@dataclass
class MirroredTeleoperateConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    fps: int = 60
    teleop_time_s: float | None = None
    display_data: bool = False


def mirror_and_invert(leader_action: dict[str, float]) -> dict[str, float]:
    """
    Swap left/right prefixes and invert twist joints (shoulder_pan, wrist_roll).
    Example input keys (from BiSO100Leader): 'left_shoulder_pan.pos', 'right_wrist_roll.pos', 'left_gripper.pos'
    Output keys (for BiSO100Follower):      'right_shoulder_pan.pos', 'left_wrist_roll.pos', 'right_gripper.pos'
    """
    mirrored: dict[str, float] = {}

    for key, val in leader_action.items():
        if key.startswith("left_"):
            suffix = key[len("left_"):]  # e.g., 'shoulder_pan.pos'
            target_key = f"right_{suffix}"
        elif key.startswith("right_"):
            suffix = key[len("right_"):]
            target_key = f"left_{suffix}"
        else:
            # Pass-through any non-prefixed keys (unlikely in bimanual leader, but safe)
            target_key = key
            suffix = key

        # Invert twist joints
        if suffix in TWIST_JOINT_NAMES:
            val = -val

        mirrored[target_key] = float(val)

    return mirrored


def teleop_loop_mirrored(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
):
    display_len = max(len(k) for k in robot.action_features)
    t_start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Read leader action
        leader_action = teleop.get_action()
        if not leader_action:
            precise_sleep(1 / fps)
            continue

        # Mirror and invert
        follower_action = mirror_and_invert(leader_action)

        # Send to follower
        _ = robot.send_action(follower_action)

        # Optional visualization
        if display_data:
            obs = robot.get_observation()
            log_rerun_data(observation=obs, action=follower_action)

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in follower_action.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(follower_action) + 3)

        # Timing
        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - t_start >= duration:
            return


@parser.wrap()
def teleoperate_mirrored(cfg: MirroredTeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation_mirrored")

    teleop = make_teleoperator_from_config(cfg.teleop)  # expects bi_so100_leader
    robot = make_robot_from_config(cfg.robot)           # expects bi_so100_follower

    teleop.connect()
    robot.connect()

    try:
        teleop_loop_mirrored(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    teleoperate_mirrored()


if __name__ == "__main__":
    main()