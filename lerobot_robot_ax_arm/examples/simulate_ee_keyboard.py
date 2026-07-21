#!/usr/bin/env python

"""Hardware-free 3D simulation of the AX-arm keyboard EE teleop.

Reuses the real IK (``_build_kinematics`` / ``_joint_velocity`` from ``teleoperate_ee_keyboard``)
and a synthetic calibration, driving a simulated (ideal) servo bus instead of a real one. Lets you
sanity-check the solver/frame behaviour and joint limits in a matplotlib window.

Controls (focus the plot window):
    - w / s : +X / -X       - a / d : +Y / -Y       - r / f : +Z / -Z
    - o / c : open / close gripper
    - t     : toggle global <-> local frame
    - m     : toggle dq <-> pos solver
    - q / esc : quit

Run:
    python examples/simulate_ee_keyboard.py [--solver pos] [--frame global] [--speed 0.06]
"""

import argparse
import importlib.util
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from lerobot.motors import MotorCalibration
from lerobot_robot_ax_arm.urdf_mapping import (
    ARM_JOINTS,
    REFERENCE_URDF_DEG,
    SCALE,
    URDF_LIMITS_DEG,
    ticks_to_urdf_vector,
    urdf_vector_to_ticks,
)

# Reuse the real teleop IK helpers without duplicating them.
_EE = importlib.util.spec_from_file_location(
    "_ee_teleop", str(Path(__file__).with_name("teleoperate_ee_keyboard.py"))
)
ee = importlib.util.module_from_spec(_EE)
_EE.loader.exec_module(ee)

TICK_REF = 512  # tick chosen to sit at each joint's URDF reference angle
GRIP_RANGE = (350, 600)


def _synthetic_calibration() -> dict[str, MotorCalibration]:
    """Calibration consistent with urdf_mapping: reference tick + travel limits per joint."""
    calib: dict[str, MotorCalibration] = {}
    for i, j in enumerate(ARM_JOINTS):
        lo_deg, hi_deg = URDF_LIMITS_DEG[j]
        ticks = sorted(int(round(TICK_REF + (d - REFERENCE_URDF_DEG[j]) / SCALE)) for d in (lo_deg, hi_deg))
        calib[j] = MotorCalibration(id=i + 1, drive_mode=0, homing_offset=TICK_REF,
                                    range_min=ticks[0], range_max=ticks[1])
    calib["gripper"] = MotorCalibration(id=4, drive_mode=0, homing_offset=0,
                                        range_min=GRIP_RANGE[0], range_max=GRIP_RANGE[1])
    return calib


class _SimBus:
    """Ideal servo bus: Present_Position instantly follows the last commanded Goal_Position."""

    def __init__(self, ticks: dict[str, float]):
        self.ticks = ticks

    def read(self, _reg, motor, normalize=False):
        return self.ticks[motor]

    def write(self, _reg, motor, value, normalize=False):
        self.ticks[motor] = float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=ee.CART_STEP_M)
    parser.add_argument("--frame", choices=("local", "global"), default="local")
    parser.add_argument("--solver", choices=("dq", "pos"), default="dq")
    args = parser.parse_args()

    from importlib.resources import files
    urdf_path = str(files("lerobot_robot_ax_arm") / "urdf" / "ax_arm.urdf")
    kin = ee._build_kinematics(urdf_path)
    link_fks = ee.build_link_fks(urdf_path)
    calib = _synthetic_calibration()

    q_start_deg = np.array([REFERENCE_URDF_DEG[j] for j in ARM_JOINTS])  # reference "zero" pose (0, 45, 90)
    start_ticks = urdf_vector_to_ticks(q_start_deg, calib)
    ticks = {j: float(start_ticks[j]) for j in ARM_JOINTS}
    ticks["gripper"] = float(sum(GRIP_RANGE) / 2)
    bus = _SimBus(ticks)

    pending = {"x": 0.0, "y": 0.0, "z": 0.0, "g": 0.0}
    state = {"frame": args.frame, "solver": args.solver}
    keymap = {"w": ("x", 1), "s": ("x", -1), "a": ("y", 1), "d": ("y", -1), "r": ("z", 1), "f": ("z", -1),
              "o": ("g", 1), "c": ("g", -1)}

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    (chain_line,) = ax.plot([], [], [], "-o", lw=3, color="tab:blue")
    (ee_pt,) = ax.plot([], [], [], "o", ms=10, color="tab:red")
    reach = 0.32
    ax.set_xlim(-reach, reach); ax.set_ylim(-reach, reach); ax.set_zlim(-0.05, reach)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    def on_key(event):
        k = (event.key or "").lower()
        if k in ("q", "escape"):
            plt.close(fig)
        elif k == "t":
            state["frame"] = "global" if state["frame"] == "local" else "local"
        elif k == "m":
            state["solver"] = "pos" if state["solver"] == "dq" else "dq"
        elif k in keymap:
            axis, direction = keymap[k]
            pending[axis] += direction

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame):
        q_rad = np.deg2rad(ticks_to_urdf_vector({j: ticks[j] for j in ARM_JOINTS}, calib))
        cmd = np.array([np.sign(pending["x"]), np.sign(pending["y"]), np.sign(pending["z"])], dtype=float)
        pending["x"] = pending["y"] = pending["z"] = 0.0

        if np.any(cmd):
            q_dot = ee._joint_velocity(kin, q_rad, cmd, state["frame"], state["solver"])
            ee_speed = float(np.linalg.norm(np.array(kin["pos_jac"](q_rad)) @ q_dot))
            if ee_speed > 1e-6:
                q_step = q_dot * (args.speed / ee_speed)
                step_norm = np.linalg.norm(q_step)
                if step_norm > ee.MAX_JOINT_STEP_RAD:
                    q_step *= ee.MAX_JOINT_STEP_RAD / step_norm
                q_target = np.clip(q_rad + q_step, kin["lower"], kin["upper"])
                target_ticks = urdf_vector_to_ticks(np.rad2deg(q_target), calib)
                for j in ARM_JOINTS:
                    c = calib[j]
                    bus.write("Goal_Position", j, int(np.clip(target_ticks[j], c.range_min, c.range_max)))

        g = np.sign(pending["g"]); pending["g"] = 0.0
        if g:
            gc = calib["gripper"]
            bus.write("Goal_Position", "gripper",
                      int(np.clip(ticks["gripper"] + g * ee.GRIP_STEP_TICK, gc.range_min, gc.range_max)))

        pts = ee.chain_points(link_fks, np.deg2rad(ticks_to_urdf_vector({j: ticks[j] for j in ARM_JOINTS}, calib)))
        chain_line.set_data(pts[:, 0], pts[:, 1]); chain_line.set_3d_properties(pts[:, 2])
        ee_pt.set_data(pts[-1:, 0], pts[-1:, 1]); ee_pt.set_3d_properties(pts[-1:, 2])
        grip_frac = (ticks["gripper"] - GRIP_RANGE[0]) / (GRIP_RANGE[1] - GRIP_RANGE[0])
        ax.set_title(f"frame={state['frame']}  solver={state['solver']}  gripper={grip_frac:.0%}\n"
                     f"w/s/a/d/r/f=move  o/c=grip  t=frame  m=solver  q=quit")
        return chain_line, ee_pt

    anim = FuncAnimation(fig, update, interval=int(1000 / ee.FPS), blit=False, cache_frame_data=False)
    fig._anim = anim  # keep a reference so it isn't garbage-collected
    plt.show()


if __name__ == "__main__":
    main()
