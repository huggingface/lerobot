"""
run.py

Demo: Teleoperate the Meca500 with a Phantom Omni haptic device
"""

import time
import mecaconfig
from meca import Meca
from lerobot.teleoperators.omni.omni import OmniTeleoperator, OmniConfig



def main():
    # --- Configs ---
    meca_cfg = mecaconfig.MecaConfig(ip="192.168.0.100")  # adjust IP if different
    omni_cfg = OmniConfig(scale_translation=.4, scale_rotation=1)

    # --- Init devices ---
    robot = Meca(meca_cfg)
    teleop = OmniTeleoperator(omni_cfg)

    try:
        robot.connect()
        teleop.connect()

        print("🤖 Teleop loop started. Press Ctrl+C to quit.")

        while True:
            observation = robot.get_observation()
            action = teleop.get_action()
            robot.send_action(action)


    except KeyboardInterrupt:
        print("🛑 Stopping teleoperation.")
    finally:
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
