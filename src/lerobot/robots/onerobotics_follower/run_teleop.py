import sys

# 导入即注册
import lerobot.teleoperators.onerobotics_leader.config_onero_arm_teleoperate  # noqa: F401
import lerobot.robots.onerobotics_follower.config_onero_arm  # noqa: F401

from lerobot.scripts.lerobot_teleoperate import main

if __name__ == "__main__":
    sys.exit(main())