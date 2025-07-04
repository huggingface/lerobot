import sys
import time

import numpy as np
import pinocchio as pin

from lerobot.robots.so101_follower_torque.config_so101_follower_t import SO101FollowerTConfig
from lerobot.robots.so101_follower_torque.so101_follower_t import SO101FollowerT

MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

FLIP_TORQUE = np.array(
    [
        True,  # shoulder_pan
        True,  # shoulder_lift   t
        True,  # elbow_flex  t
        True,  # wrist_flex  t
        True,  # wrist_roll
        False,  # gripper
    ],
    dtype=bool,
)

SIGN = np.where(FLIP_TORQUE, -1.0, 1.0)

IDX = {name: i for i, name in enumerate(MOTORS)}


def obs_to_q(obs):
    """Extract joint vector q [rad] directly from observation dict."""
    q = np.zeros(len(MOTORS))
    for m in MOTORS:
        q[IDX[m]] = obs[f"{m}.pos"]  # already in rad
    return q


def tau_to_action(tau):
    """Convert τ array (Nm) to action dict understood by send_action()."""
    tau = SIGN * tau
    return {f"{m}.effort": float(tau[IDX[m]]) for m in MOTORS}


cfg = SO101FollowerTConfig(port="/dev/tty.usbmodem58760431551", id="follower_torque6")
real = SO101FollowerT(cfg)
real.connect()

real.pin_robot.data = real.pin_robot.model.createData()
real.pin_robot.initViewer(open=True)
real.pin_robot.loadViewerModel()
real.pin_robot.display(real.pin_robot.q0)

ESC_CLR_EOL = "\x1b[K"

print("Running gravity compensation")
try:
    while True:
        obs = real.get_observation()  # positions in rad
        q = obs_to_q(obs)  # dict to vector

        tau = pin.computeGeneralizedGravity(real.pin_robot.model, real.pin_robot.data, q)  # τ in [Nm]

        # build compact single-line status
        status = " | ".join(
            f"{m}: {np.degrees(q[i]):+5.1f}° Δτ {(abs(obs[f'{m}.effort']) - abs(tau[i])):+5.1f}"
            for i, m in enumerate(MOTORS)
        )

        # overwrite previous status
        sys.stdout.write("\r" + ESC_CLR_EOL + status)
        sys.stdout.flush()

        real.send_action(tau_to_action(tau))  # apply τ

        real.pin_robot.display(q)
        time.sleep(0.002)  # Run at 500 Hz
except KeyboardInterrupt:
    print("Stopping")
finally:
    real.disconnect()
