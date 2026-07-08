#!/usr/bin/env python

"""Keyboard end-effector teleoperation for the 4-DoF AX arm (position-only IK on the URDF).

The arm has 3 revolute joints (base yaw + shoulder/elbow pitch) giving exactly 3 task-space DoF,
all spent on reaching a 3D *position* (orientation is not controllable). Each frame we:

  1. read the raw motor ticks and map them to URDF joint degrees (see ``urdf_mapping``),
  2. forward-kinematics to the current end-effector pose,
  3. offset the target position by the pressed keys,
  4. solve position-only IK (``orientation_weight=0.0``) for new URDF joint degrees,
  5. map back to ticks and command the servos.

The tick<->URDF mapping is established once by ``lerobot-calibrate`` (reference pose + travel
limits), so no separate alignment step is needed here.

Controls (letter keys; hold to keep moving via terminal key-repeat):
    - w / s : +X / -X   (forward / back)
    - a / d : +Y / -Y   (left / right)
    - r / f : +Z / -Z   (up / down)
    - o / c : open / close gripper
    - ESC / q : stop

Run:
    python examples/teleoperate_ee_keyboard.py --port /dev/tty.usbserial-XXXX --id my_ax_arm
"""

import argparse
import time
from importlib.resources import files

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.keyboard_input import create_key_listener
from lerobot.utils.robot_utils import precise_sleep

from lerobot_robot_ax_arm import AXArm, AXArmConfig
from lerobot_robot_ax_arm.urdf_mapping import (
    ARM_JOINTS,
    URDF_JOINT_NAMES,
    ticks_to_urdf_vector,
    urdf_vector_to_ticks,
)

FPS = 30
LINEAR_STEP_M = 0.005  # EE position change per pressed frame
GRIP_STEP_TICK = 15  # gripper ticks per press


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port of the AX arm")
    parser.add_argument("--id", default="my_ax_arm", help="Robot id used for calibration files")
    args = parser.parse_args()

    robot = AXArm(AXArmConfig(port=args.port, id=args.id, use_degrees=True))
    robot.connect(calibrate=False)
    if not robot.calibration:
        raise RuntimeError(f"No calibration found for id '{args.id}'. Run lerobot-calibrate first.")

    urdf_path = str(files("lerobot_robot_ax_arm") / "urdf" / "ax_arm.urdf")
    kin = RobotKinematics(urdf_path, target_frame_name="gripper_link", joint_names=URDF_JOINT_NAMES)

    grip_calib = robot.calibration["gripper"]
    grip_tick = int(robot.bus.read("Present_Position", "gripper", normalize=False))

    pending = {"x": 0.0, "y": 0.0, "z": 0.0, "g": 0.0}
    state = {"quit": False}
    keymap = {"w": ("x", 1), "s": ("x", -1), "a": ("y", 1), "d": ("y", -1), "r": ("z", 1), "f": ("z", -1),
              "o": ("g", 1), "c": ("g", -1)}

    def on_key(name: str) -> None:
        k = name.lower()
        if k in ("esc", "q"):
            state["quit"] = True
        elif k in keymap:
            axis, direction = keymap[k]
            pending[axis] += direction

    listener = create_key_listener(on_key, controls_help="w/s a/d r/f = XYZ, o/c = gripper, esc = stop")
    if listener is None:
        raise RuntimeError("Needs an interactive terminal with a usable key listener.")

    print("Keyboard EE teleop. w/s=X a/d=Y r/f=Z o/c=gripper, ESC=stop.")
    try:
        while not state["quit"]:
            t0 = time.perf_counter()

            ticks = {j: float(robot.bus.read("Present_Position", j, normalize=False)) for j in ARM_JOINTS}
            q_deg = ticks_to_urdf_vector(ticks, robot.calibration)

            pose = kin.forward_kinematics(q_deg)
            dx, dy, dz = (np.sign(pending[a]) for a in ("x", "y", "z"))
            pending["x"] = pending["y"] = pending["z"] = 0.0
            pose[:3, 3] += LINEAR_STEP_M * np.array([dx, dy, dz])

            q_target = kin.inverse_kinematics(q_deg, pose, orientation_weight=0.0)
            target_ticks = urdf_vector_to_ticks(q_target, robot.calibration)
            for j in ARM_JOINTS:
                c = robot.calibration[j]
                tick = int(np.clip(target_ticks[j], c.range_min, c.range_max))
                robot.bus.write("Goal_Position", j, tick, normalize=False)

            g = np.sign(pending["g"])
            pending["g"] = 0.0
            if g:
                grip_tick = int(np.clip(grip_tick + g * GRIP_STEP_TICK, grip_calib.range_min, grip_calib.range_max))
                robot.bus.write("Goal_Position", "gripper", grip_tick, normalize=False)

            urdf_str = " ".join(f"{j}={v:+6.1f}" for j, v in zip(ARM_JOINTS, q_target))
            print(f"cmd[x={dx:+.0f} y={dy:+.0f} z={dz:+.0f} g={g:+.0f}] -> urdf[{urdf_str}] gripper={grip_tick}",
                  end="\r", flush=True)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        print()
        robot.disconnect()


if __name__ == "__main__":
    main()
