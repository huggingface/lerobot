import time

import numpy as np
import pinocchio as pin

from lerobot.common.robots.so101_follower_torque.config_so101_follower_t import SO101FollowerTConfig
from lerobot.common.robots.so101_follower_torque.so101_follower_t import SO101FollowerT

MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

IDX = {name: i for i, name in enumerate(MOTORS)}


def obs_to_q(obs):
    """Extract joint vector q [rad] directly from observation dict."""
    q = np.zeros(len(MOTORS))
    for m in MOTORS:
        q[IDX[m]] = obs[f"{m}.pos"]  # already in rad
    return q


def tau_to_action(tau):
    """Convert τ array (Nm) to action dict understood by send_action()."""
    return {f"{m}.effort": float(tau[IDX[m]]) for m in MOTORS}


pin_robot = pin.RobotWrapper.BuildFromURDF(
    "lerobot/SO101/so101_new_calib.urdf",
    "lerobot/SO101",
)
pin_robot.data = pin_robot.model.createData()
pin_robot.initViewer(open=True)
pin_robot.loadViewerModel()
pin_robot.display(pin_robot.q0)

cfg = SO101FollowerTConfig(port="/dev/tty.usbmodem58760431551", id="follower_t")
real = SO101FollowerT(cfg)
real.connect()

print("Running gravity compensation")
try:
    while True:
        obs = real.get_observation()  # positions in rad
        q = obs_to_q(obs)  # dict to vector

        tau = pin.computeGeneralizedGravity(pin_robot.model, pin_robot.data, q)  # τ in [Nm]

        # real.send_action(tau_to_action(tau)) # apply τ

        pin_robot.display(q)
        time.sleep(0.02)  # Run at 50 Hz
except KeyboardInterrupt:
    print("Stopping")
finally:
    real.disconnect()
