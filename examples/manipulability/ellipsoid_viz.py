#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Real-time 3D manipulability ellipsoid visualization for the SO-101 arm.

Connects to the follower arm in observe mode (torque OFF), reads joint
positions, computes the translational Jacobian, and renders the velocity
manipulability ellipsoid live in a matplotlib 3D window.

The ellipsoid axes are the left singular vectors of Jv, scaled by the
corresponding singular values.  A flat/degenerate ellipsoid means the
arm is near singularity (it can't move well in one direction).

Color coding:
  Green  = OK        (σ_min ≥ 0.015)
  Yellow = WARNING   (0.006 ≤ σ_min < 0.015)
  Red    = CRITICAL  (σ_min < 0.006)

Example usage:

    python examples/manipulability/ellipsoid_viz.py \
        --robot.port=/dev/ttyACM0 \
        --robot.id=my_follower \
        --urdf=path/to/so101.urdf
"""

import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from lerobot.model.kinematics import RobotKinematics
from lerobot.model.manipulability import extract_translational_jacobian
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

SIGMA_MIN_WARN = 0.015
SIGMA_MIN_CRITICAL = 0.006

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
ARM_JOINT_NAMES = MOTOR_NAMES[:5]

N_PHI = 20
N_THETA = 20
SCALE = 10.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time 3D manipulability ellipsoid visualization"
    )
    parser.add_argument("--robot.port", dest="robot_port", required=True, help="Follower serial port")
    parser.add_argument("--robot.id", dest="robot_id", required=True, help="Follower calibration ID")
    parser.add_argument("--urdf", required=True, help="Path to SO-101 URDF file")
    parser.add_argument("--scale", type=float, default=SCALE, help="Ellipsoid display scale (default: 10)")
    parser.add_argument("--interval", type=int, default=100, help="Update interval in ms (default: 100)")
    return parser.parse_args()


def make_unit_sphere():
    """Return (x, y, z) arrays for a unit sphere mesh."""
    phi = np.linspace(0, 2 * np.pi, N_PHI)
    theta = np.linspace(0, np.pi, N_THETA)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def main():
    args = parse_args()
    scale = args.scale

    # ---- Connect to arm ------------------------------------------------------
    follower = SO101Follower(
        SO101FollowerConfig(port=args.robot_port, id=args.robot_id, use_degrees=False)
    )
    follower.connect()
    follower.bus.disable_torque()
    print("Connected — torque OFF.  Move the arm by hand.")

    # ---- Kinematics ----------------------------------------------------------
    kin = RobotKinematics(args.urdf, "gripper_frame_link", MOTOR_NAMES)
    limits_deg = {}
    for name in ARM_JOINT_NAMES:
        lo, hi = kin.robot.get_joint_limits(name)
        limits_deg[name] = (np.rad2deg(lo), np.rad2deg(hi))

    def norm_to_deg(obs):
        q = np.zeros(len(MOTOR_NAMES))
        for i, m in enumerate(MOTOR_NAMES):
            v = float(obs.get(f"{m}.pos", 0.0))
            if m in limits_deg:
                lo, hi = limits_deg[m]
                q[i] = lo + (v + 100.0) / 200.0 * (hi - lo)
        return q

    # ---- Unit sphere for ellipsoid -------------------------------------------
    sx, sy, sz = make_unit_sphere()
    sphere_pts = np.stack([sx.ravel(), sy.ravel(), sz.ravel()], axis=0)

    # ---- Set up matplotlib figure --------------------------------------------
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("SO-101 Manipulability Ellipsoid", fontsize=16, fontweight="bold")

    surf = [ax.plot_surface(sx, sy, sz, alpha=0.6, color="green")]

    info_text = ax.text2D(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=13,
        fontfamily="monospace", verticalalignment="top",
    )
    sigma_text = ax.text2D(
        0.02, 0.82, "", transform=ax.transAxes, fontsize=11,
        fontfamily="monospace", verticalalignment="top",
    )

    arrow_artists = []

    axis_lim = 0.25 * scale
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)
    ax.set_xlabel("X (forward)", fontsize=10)
    ax.set_ylabel("Y (left)", fontsize=10)
    ax.set_zlabel("Z (up)", fontsize=10)
    ax.set_box_aspect([1, 1, 1])

    def update(frame_num):
        nonlocal surf, arrow_artists

        obs = follower.get_observation()
        q_deg = norm_to_deg(obs)

        j_arm = kin.compute_frame_jacobian(q_deg, joint_names=ARM_JOINT_NAMES)
        jv = extract_translational_jacobian(j_arm)
        u, s, _vt = np.linalg.svd(jv, full_matrices=False)

        sigma_min = s.min()
        sigma_max = s.max()
        cond = sigma_max / sigma_min if sigma_min > 1e-10 else float("inf")

        if sigma_min < SIGMA_MIN_CRITICAL:
            color, status, edge_color = "#ff3333", "▓▓ CRITICAL", "#ff0000"
        elif sigma_min < SIGMA_MIN_WARN:
            color, status, edge_color = "#ffcc00", "▒▒ WARNING", "#ff8800"
        else:
            color, status, edge_color = "#33ff88", "░░ OK", "#00cc44"

        ellipsoid_pts = u @ np.diag(s * scale) @ sphere_pts
        ex = ellipsoid_pts[0].reshape(sx.shape)
        ey = ellipsoid_pts[1].reshape(sy.shape)
        ez = ellipsoid_pts[2].reshape(sz.shape)

        surf[0].remove()
        surf[0] = ax.plot_surface(
            ex, ey, ez, alpha=0.45, color=color,
            edgecolor=edge_color, linewidth=0.3,
        )

        for a in arrow_artists:
            a.remove()
        arrow_artists.clear()

        axis_colors = ["#ff4444", "#44ff44", "#4488ff"]
        for i in range(3):
            direction = u[:, i] * s[i] * scale
            arrow = ax.quiver(
                0, 0, 0, direction[0], direction[1], direction[2],
                color=axis_colors[i], linewidth=2.5, arrow_length_ratio=0.12,
            )
            arrow_artists.append(arrow)

        info_text.set_text(f"{status}")
        info_text.set_color(color)

        sigma_text.set_text(
            f"σ₁={s[0]:.4f}  σ₂={s[1]:.4f}  σ₃={s[2]:.4f}\n"
            f"σ_min={sigma_min:.4f}  cond={cond:.1f}"
        )
        sigma_text.set_color("white")

        return [surf[0], info_text, sigma_text] + arrow_artists

    _ani = animation.FuncAnimation(
        fig, update, interval=args.interval, blit=False, cache_frame_data=False,
    )

    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        follower.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
