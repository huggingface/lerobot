#!/usr/bin/env python
"""
Teleoperation script for bi_so_follower_mobile with rudder pedal wheel control.

Requires these plugins to be installed:
  pip install -e /path/to/lerobot_robot_bi_so_follower_mobile
  pip install -e /path/to/lerobot_teleoperator_rudder_pedal

Usage:
  python -m lerobot.scripts.teleop_bimanual_mobile \
    --robot.type=bi_so_follower_mobile \
    --robot.left_arm_config.port=/dev/tty.usbmodem5B3D0422841 \
    --robot.right_arm_config.port=/dev/tty.usbmodem5AE60849171 \
    --robot.id=bi_so101_follower \
    --arm_teleop.type=bi_so_leader \
    --arm_teleop.left_arm_config.port=/dev/tty.usbmodem5B3D0466431 \
    --arm_teleop.right_arm_config.port=/dev/tty.usbmodem5AE60531711 \
    --arm_teleop.id=bi_so101_leader \
    --wheel_teleop.max_speed=500.0
"""

import logging
import time
from dataclasses import dataclass, field

import draccus

from lerobot.robots.config import RobotConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config

# These imports trigger register_subclass decorators
import lerobot_robot_bi_so_follower_mobile  # noqa: F401
import lerobot_teleoperator_rudder_pedal    # noqa: F401
# bi_so_leader lives inside lerobot itself (not a plugin package) so it does not
# get auto-discovered by the plugin prefix system. We import it explicitly here
# to trigger its @TeleoperatorConfig.register_subclass decorator so draccus can
# parse --arm_teleop.type=bi_so_leader from the command line.
from lerobot.teleoperators import bi_so_leader  # noqa: F401
from lerobot_teleoperator_rudder_pedal import RudderPedal, RudderPedalConfig

logger = logging.getLogger(__name__)


@dataclass
class TeleopBimanualMobileConfig:
    robot: RobotConfig
    arm_teleop: TeleoperatorConfig
    wheel_teleop: RudderPedalConfig = field(default_factory=RudderPedalConfig)
    fps: int = 60
    display_data: bool = False


@draccus.wrap()
def main(cfg: TeleopBimanualMobileConfig):
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    robot        = make_robot_from_config(cfg.robot)
    arm_teleop   = make_teleoperator_from_config(cfg.arm_teleop)
    wheel_teleop = RudderPedal(cfg.wheel_teleop)

    try:
        robot.connect()
        arm_teleop.connect()
        wheel_teleop.connect()

        print("Teleoperation started.")
        print("  Right brake = forward | Left brake = reverse | Rudder = turn")
        print("  Ctrl+C to stop.")

        dt = 1.0 / cfg.fps

        while True:
            start = time.perf_counter()

            arm_action   = arm_teleop.get_action()
            wheel_action = wheel_teleop.get_action()
            action       = {**arm_action, **wheel_action}

            robot.send_action(action)

            if cfg.display_data:
                print(
                    f"L_wheel: {wheel_action['base_left_wheel.vel']:+6.1f}  "
                    f"R_wheel: {wheel_action['base_right_wheel.vel']:+6.1f}",
                    end="\r"
                )

            elapsed = time.perf_counter() - start
            time.sleep(max(0, dt - elapsed))

    except KeyboardInterrupt:
        print("\nStopping teleoperation.")
    finally:
        robot.disconnect()
        arm_teleop.disconnect()
        wheel_teleop.disconnect()


if __name__ == "__main__":
    main()
