#!/usr/bin/env python

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
Unitree G1 teleoperator using bimanual exoskeleton arms with IK.

Handles serial communication with exoskeleton hardware and converts
raw ADC sensor readings to joint angles using ellipse-fit calibration.

Uses inverse kinematics: exoskeleton FK computes end-effector pose,
G1 IK solves for joint angles.
"""

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import serial
from huggingface_hub import snapshot_download

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex, G1_29_JointIndex
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS

from ..teleoperator import Teleoperator
from .config_unitree_g1 import UnitreeG1TeleoperatorConfig

logger = logging.getLogger(__name__)

# =============================================================================
# EXOSKELETON CONSTANTS
# =============================================================================

SENSOR_COUNT = 16
ADC_MAX = 4095
CENTER_GUESS = 1950.0

# Joint configuration: (name, sensor_pair, flipped)
JOINTS = [
    ("shoulder_pitch", (0, 1), True),
    ("shoulder_yaw", (2, 3), True),
    ("shoulder_roll", (4, 5), False),
    ("elbow_flex", (6, 7), True),
    ("wrist_roll", (14, 15), False),
]


# =============================================================================
# EXOSKELETON CALIBRATION DATA CLASSES
# =============================================================================

@dataclass
class ExoskeletonJointCalibration:
    """Calibration data for a single exoskeleton joint."""
    name: str
    pair: tuple[int, int]
    flipped: bool
    center_guess: float
    center_fit: list[float]
    T: list[list[float]]  # 2x2 transformation matrix
    zero_offset: float = 0.0


@dataclass
class ExoskeletonCalibration:
    """Full calibration data for an exoskeleton arm."""
    version: int = 2
    side: str = ""
    adc_max: int = ADC_MAX
    created_unix: float = 0.0
    joints: list[ExoskeletonJointCalibration] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "version": self.version,
            "side": self.side,
            "adc_max": self.adc_max,
            "created_unix": self.created_unix,
            "joints": [
                {
                    "name": j.name,
                    "pair": list(j.pair),
                    "flipped": j.flipped,
                    "center_guess": j.center_guess,
                    "center_fit": j.center_fit,
                    "T": j.T,
                    "zero_offset": j.zero_offset,
                }
                for j in self.joints
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExoskeletonCalibration":
        """Load from dict."""
        joints = [
            ExoskeletonJointCalibration(
                name=j["name"],
                pair=tuple(j["pair"]),
                flipped=j["flipped"],
                center_guess=j.get("center_guess", CENTER_GUESS),
                center_fit=j["center_fit"],
                T=j["T"],
                zero_offset=j.get("zero_offset", 0.0),
            )
            for j in data.get("joints", [])
        ]
        return cls(
            version=data.get("version", 2),
            side=data.get("side", ""),
            adc_max=data.get("adc_max", ADC_MAX),
            created_unix=data.get("created_unix", 0.0),
            joints=joints,
        )


# =============================================================================
# EXOSKELETON UTILITIES
# =============================================================================

def parse_raw16(line: str) -> list[int] | None:
    """Parse a line of 16 space-separated ADC values."""
    parts = line.strip().split()
    if len(parts) < SENSOR_COUNT:
        return None
    try:
        return [int(x) for x in parts[:SENSOR_COUNT]]
    except ValueError:
        return None


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def exo_raw_to_angles(raw16: list[int], calib: ExoskeletonCalibration) -> dict[str, float]:
    """
    Convert raw sensor readings to joint angles using calibration.

    Args:
        raw16: List of 16 raw ADC values
        calib: Calibration data

    Returns:
        Dict mapping joint name to angle in radians
    """
    angles = {}
    for jdata in calib.joints:
        pair = jdata.pair
        flipped = jdata.flipped
        center_fit = np.array(jdata.center_fit)
        T = np.array(jdata.T)
        zero_offset = jdata.zero_offset

        s = raw16[pair[0]]
        c = raw16[pair[1]]
        if flipped:
            s = ADC_MAX - s
            c = ADC_MAX - c

        x_raw = float(c) - CENTER_GUESS
        y_raw = float(s) - CENTER_GUESS

        p = np.array([x_raw, y_raw], dtype=float)
        z = T @ (p - center_fit)
        angle = float(np.arctan2(z[1], z[0])) - zero_offset

        # Negate and normalize
        angles[jdata.name] = normalize_angle(-angle)

    return angles


# =============================================================================
# EXOSKELETON ARM CLASS
# =============================================================================

class ExoskeletonArm:
    """
    Reads raw sensor data from an exoskeleton arm over serial
    and converts to joint angles using calibration.
    """

    def __init__(self, port: str, baud_rate: int = 115200, calibration_fpath: Path | None = None, side: str = ""):
        self.port = port
        self.baud_rate = baud_rate
        self.side = side
        self.calibration_fpath = calibration_fpath
        self.calibration: ExoskeletonCalibration | None = None
        
        self._ser: serial.Serial | None = None
        self._connected = False
        self._calibrated = False

        # Load calibration if file exists
        if calibration_fpath and calibration_fpath.is_file():
            self._load_calibration()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated and self.calibration is not None

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the exoskeleton serial port."""
        if self._connected:
            return

        try:
            self._ser = serial.Serial(self.port, self.baud_rate, timeout=0.02)
            self._ser.reset_input_buffer()
            self._connected = True
            logger.info(f"Connected to exoskeleton at {self.port}")
        except serial.SerialException as e:
            raise ConnectionError(f"Failed to connect to {self.port}: {e}")

        if calibrate and not self.is_calibrated:
            self.calibrate()

    def disconnect(self) -> None:
        """Disconnect from the serial port."""
        if self._ser is not None:
            self._ser.close()
            self._ser = None
        self._connected = False

    def _load_calibration(self) -> None:
        """Load calibration from file."""
        if self.calibration_fpath is None:
            return
        try:
            with open(self.calibration_fpath) as f:
                data = json.load(f)
            self.calibration = ExoskeletonCalibration.from_dict(data)
            self._calibrated = True
            logger.info(f"Loaded calibration from {self.calibration_fpath}")
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}")
            self._calibrated = False

    def _save_calibration(self) -> None:
        """Save calibration to file."""
        if self.calibration_fpath is None or self.calibration is None:
            return
        self.calibration_fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.calibration_fpath, "w") as f:
            json.dump(self.calibration.to_dict(), f, indent=2)
        logger.info(f"Saved calibration to {self.calibration_fpath}")

    def read_raw(self) -> list[int] | None:
        """Read latest raw16 sample, draining the buffer."""
        if self._ser is None:
            return None

        last = None
        while self._ser.in_waiting > 0:
            b = self._ser.readline()
            if not b:
                break
            raw16 = parse_raw16(b.decode("utf-8", errors="ignore"))
            if raw16 is not None:
                last = raw16

        if last is None:
            b = self._ser.readline()
            if b:
                last = parse_raw16(b.decode("utf-8", errors="ignore"))

        return last

    def get_angles(self) -> dict[str, float]:
        """Get current joint angles in radians."""
        if not self.is_calibrated or self.calibration is None:
            raise RuntimeError("Exoskeleton not calibrated")

        raw = self.read_raw()
        if raw is None:
            return {}

        return exo_raw_to_angles(raw, self.calibration)

    def calibrate(self) -> None:
        """
        Run interactive calibration for the exoskeleton arm.
        
        This opens a matplotlib window for ellipse fitting and zero-pose capture.
        """
        try:
            import cv2
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "Calibration requires matplotlib and opencv-python. "
                "Install with: pip install matplotlib opencv-python"
            ) from e

        if not self._connected or self._ser is None:
            raise RuntimeError("Must be connected before calibrating")

        logger.info(f"Starting calibration for {self.side} exoskeleton arm")

        # Calibration parameters
        FIT_EVERY = 0.15
        MIN_FIT_POINTS = 60
        FIT_WINDOW = 900
        MAX_FIT_POINTS = 300
        TRIM_LOW = 0.05
        TRIM_HIGH = 0.95
        MEDIAN_WINDOW = 5
        HISTORY = 3500
        DRAW_HZ = 120.0
        SAMPLE_COUNT = 50

        def running_median(win: deque) -> float:
            return float(np.median(np.fromiter(win, dtype=float)))

        def read_joint_point(raw16: list[int], pair: tuple[int, int], flipped: bool, center: float):
            s = raw16[pair[0]]
            c = raw16[pair[1]]
            if flipped:
                s = ADC_MAX - s
                c = ADC_MAX - c
            return float(c) - center, float(s) - center, float(s), float(c)

        def select_fit_subset(xs, ys):
            n_all = len(xs)
            n = min(FIT_WINDOW, n_all)
            if n <= 0:
                return None, None

            x = np.asarray(list(xs)[-n:], dtype=float)
            y = np.asarray(list(ys)[-n:], dtype=float)

            r = np.sqrt(x * x + y * y)
            if len(r) >= 20:
                lo = np.quantile(r, TRIM_LOW)
                hi = np.quantile(r, TRIM_HIGH)
                keep = (r >= lo) & (r <= hi)
                x = x[keep]
                y = y[keep]

            if len(x) > MAX_FIT_POINTS:
                idx = np.linspace(0, len(x) - 1, MAX_FIT_POINTS).astype(int)
                x = x[idx]
                y = y[idx]

            return x, y

        def fit_ellipse_opencv(x, y):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            if len(x) < 5:
                return None

            pts = np.stack([x, y], axis=1).astype(np.float32).reshape(-1, 1, 2)
            try:
                (xc, yc), (w, h), angle_deg = cv2.fitEllipse(pts)
            except cv2.error:
                return None

            a = float(w) * 0.5
            b = float(h) * 0.5
            phi = np.deg2rad(float(angle_deg))

            if b > a:
                a, b = b, a
                phi = phi + np.pi / 2.0

            if not np.isfinite(a) or not np.isfinite(b) or a <= 1e-6 or b <= 1e-6:
                return None

            cp = float(np.cos(phi))
            sp = float(np.sin(phi))
            R = np.array([[cp, -sp], [sp, cp]], dtype=float)

            center = np.array([float(xc), float(yc)], dtype=float)

            tt = np.linspace(0, 2 * np.pi, 360)
            circ = np.stack([a * np.cos(tt), b * np.sin(tt)], axis=0)
            outline = (R @ circ).T + center[None, :]

            return {
                "center": center,
                "a": a,
                "b": b,
                "R": R,
                "ex": outline[:, 0],
                "ey": outline[:, 1],
            }

        # Setup matplotlib
        plt.ion()
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))

        ax0.set_xlabel("cos - center")
        ax0.set_ylabel("sin - center")
        ax0.grid(True, alpha=0.25)
        ax0.set_aspect("equal", adjustable="box")

        ax1.set_title("Unit circle + angle")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.grid(True, alpha=0.25)
        ax1.set_aspect("equal", adjustable="box")

        tt = np.linspace(0, 2 * np.pi, 360)
        ax1.plot(np.cos(tt), np.sin(tt), "k-", linewidth=1)

        ax0.set_xlim(-2200, 2200)
        ax0.set_ylim(-2200, 2200)
        ax1.set_xlim(-1.4, 1.4)
        ax1.set_ylim(-1.4, 1.4)

        sc0 = ax0.scatter([], [], s=6, animated=True)
        (ell_line,) = ax0.plot([], [], "r-", linewidth=2, animated=True)
        sc1 = ax1.scatter([], [], s=6, animated=True)
        (radius_line,) = ax1.plot([], [], "g-", linewidth=2, animated=True)
        angle_text = ax1.text(0.02, 0.98, "", transform=ax1.transAxes,
                              va="top", ha="left", fontsize=12, animated=True)

        fig.canvas.draw()
        bg0 = fig.canvas.copy_from_bbox(ax0.bbox)
        bg1 = fig.canvas.copy_from_bbox(ax1.bbox)

        # State
        joints_out = []
        joint_idx = 0
        phase = "ellipse"
        advance_requested = False
        zero_samples = []

        def on_key(event):
            nonlocal advance_requested
            if event.key in ("n", "N", "enter", " "):
                advance_requested = True

        fig.canvas.mpl_connect("key_press_event", on_key)

        def reset_joint_state():
            return {
                "xs": deque(maxlen=HISTORY),
                "ys": deque(maxlen=HISTORY),
                "xu": deque(maxlen=HISTORY),
                "yu": deque(maxlen=HISTORY),
                "win_s": deque(maxlen=MEDIAN_WINDOW),
                "win_c": deque(maxlen=MEDIAN_WINDOW),
                "ellipse_cache": None,
                "T": None,
                "center_fit": None,
                "have_transform": False,
                "latest_z": None,
                "last_fit": 0.0,
            }

        state = reset_joint_state()
        last_draw = 0.0

        name, pair, flipped = JOINTS[joint_idx]
        fig.canvas.manager.set_window_title(f"[{joint_idx+1}/{len(JOINTS)}] {name} - ELLIPSE")
        ax0.set_title(f"{name} raw (filtered)")
        logger.info(f"[{joint_idx+1}/{len(JOINTS)}] Calibrating {name}")
        logger.info("Step 1: Move joint around to map ellipse, then press 'n'")

        try:
            while plt.fignum_exists(fig.number):
                name, pair, flipped = JOINTS[joint_idx]

                # State machine
                if phase == "ellipse" and advance_requested and state["have_transform"]:
                    joints_out.append({
                        "name": name,
                        "pair": [int(pair[0]), int(pair[1])],
                        "flipped": bool(flipped),
                        "center_guess": float(CENTER_GUESS),
                        "center_fit": state["center_fit"].tolist(),
                        "T": state["T"].tolist(),
                    })
                    logger.info(f"  -> Ellipse saved for {name}")

                    phase = "zero_pose"
                    zero_samples = []
                    advance_requested = False

                    fig.canvas.manager.set_window_title(f"[{joint_idx+1}/{len(JOINTS)}] {name} - ZERO POSE")
                    ax0.set_title(f"{name} - hold zero pose")
                    fig.canvas.draw()
                    bg0 = fig.canvas.copy_from_bbox(ax0.bbox)
                    bg1 = fig.canvas.copy_from_bbox(ax1.bbox)
                    logger.info(f"Step 2: Hold {name} in zero position, then press 'n'")

                elif phase == "ellipse" and advance_requested and not state["have_transform"]:
                    logger.info("  (Need valid fit first - keep moving the joint)")
                    advance_requested = False

                elif phase == "zero_pose" and advance_requested:
                    if len(zero_samples) >= SAMPLE_COUNT:
                        zero_offset = float(np.mean(zero_samples[-SAMPLE_COUNT:]))
                        joints_out[-1]["zero_offset"] = zero_offset
                        logger.info(f"  -> {name} zero: {zero_offset:+.3f} rad ({np.degrees(zero_offset):+.1f}°)")

                        joint_idx += 1
                        advance_requested = False

                        if joint_idx >= len(JOINTS):
                            # All joints calibrated
                            self.calibration = ExoskeletonCalibration(
                                version=2,
                                side=self.side,
                                adc_max=ADC_MAX,
                                created_unix=time.time(),
                                joints=[
                                    ExoskeletonJointCalibration(
                                        name=j["name"],
                                        pair=tuple(j["pair"]),
                                        flipped=j["flipped"],
                                        center_guess=j["center_guess"],
                                        center_fit=j["center_fit"],
                                        T=j["T"],
                                        zero_offset=j.get("zero_offset", 0.0),
                                    )
                                    for j in joints_out
                                ],
                            )
                            self._calibrated = True
                            self._save_calibration()
                            logger.info("Calibration complete!")
                            break

                        phase = "ellipse"
                        state = reset_joint_state()
                        name, pair, flipped = JOINTS[joint_idx]
                        fig.canvas.manager.set_window_title(f"[{joint_idx+1}/{len(JOINTS)}] {name} - ELLIPSE")
                        ax0.set_title(f"{name} raw (filtered)")
                        fig.canvas.draw()
                        bg0 = fig.canvas.copy_from_bbox(ax0.bbox)
                        bg1 = fig.canvas.copy_from_bbox(ax1.bbox)
                        logger.info(f"[{joint_idx+1}/{len(JOINTS)}] Calibrating {name}")
                        logger.info("Step 1: Move joint around to map ellipse, then press 'n'")
                    else:
                        logger.info(f"  (Collecting samples: {len(zero_samples)}/{SAMPLE_COUNT} - hold still)")
                        advance_requested = False

                # Read sensor data
                raw16 = self.read_raw()
                if raw16 is not None:
                    x_raw, y_raw, s_raw, c_raw = read_joint_point(raw16, pair, flipped, CENTER_GUESS)

                    if phase == "ellipse":
                        if state["have_transform"]:
                            p = np.array([x_raw, y_raw], dtype=float)
                            z = state["T"] @ (p - state["center_fit"])
                            state["xu"].append(float(z[0]))
                            state["yu"].append(float(z[1]))
                            state["latest_z"] = (float(z[0]), float(z[1]))

                        state["win_s"].append(s_raw)
                        state["win_c"].append(c_raw)
                        if len(state["win_s"]) >= max(3, MEDIAN_WINDOW):
                            s_f = running_median(state["win_s"])
                            c_f = running_median(state["win_c"])
                            state["ys"].append(s_f - CENTER_GUESS)
                            state["xs"].append(c_f - CENTER_GUESS)

                    else:  # zero_pose
                        jdata = joints_out[-1]
                        center_fit = np.array(jdata["center_fit"], dtype=float)
                        T = np.array(jdata["T"], dtype=float)
                        p = np.array([x_raw, y_raw], dtype=float)
                        z = T @ (p - center_fit)
                        angle = float(np.arctan2(z[1], z[0]))
                        zero_samples.append(angle)
                        state["latest_z"] = (float(z[0]), float(z[1]))

                # Ellipse fitting
                t = time.time()
                if phase == "ellipse" and (t - state["last_fit"]) >= FIT_EVERY and len(state["xs"]) >= MIN_FIT_POINTS:
                    xfit, yfit = select_fit_subset(state["xs"], state["ys"])
                    if xfit is not None and len(xfit) >= MIN_FIT_POINTS:
                        fit = fit_ellipse_opencv(xfit, yfit)
                        if fit is not None:
                            state["center_fit"] = fit["center"]
                            state["T"] = np.diag([1.0 / fit["a"], 1.0 / fit["b"]]) @ fit["R"].T
                            state["ellipse_cache"] = (fit["ex"], fit["ey"])
                            state["have_transform"] = True

                    state["last_fit"] = t

                # Drawing
                if (t - last_draw) >= 1.0 / DRAW_HZ:
                    fig.canvas.restore_region(bg0)
                    fig.canvas.restore_region(bg1)

                    if phase == "ellipse":
                        sc0.set_offsets(np.c_[state["xs"], state["ys"]] if len(state["xs"]) else np.empty((0, 2)))
                        ax0.draw_artist(sc0)

                        if state["ellipse_cache"] is not None:
                            ell_line.set_data(*state["ellipse_cache"])
                        else:
                            ell_line.set_data([], [])
                        ax0.draw_artist(ell_line)

                        sc1.set_offsets(np.c_[state["xu"], state["yu"]] if len(state["xu"]) else np.empty((0, 2)))
                        ax1.draw_artist(sc1)

                        if state["latest_z"] is not None:
                            zx, zy = state["latest_z"]
                            radius_line.set_data([0.0, zx], [0.0, zy])
                            ang = float(np.arctan2(zy, zx))
                            angle_text.set_text(
                                f"angle: {ang:+.3f} rad  ({np.degrees(ang):+.1f}°)\n"
                                f"press 'n' when ellipse looks good"
                            )
                        else:
                            radius_line.set_data([], [])
                            angle_text.set_text("(waiting for fit)")

                    else:  # zero_pose
                        sc0.set_offsets(np.empty((0, 2)))
                        ax0.draw_artist(sc0)
                        ell_line.set_data([], [])
                        ax0.draw_artist(ell_line)

                        if state["latest_z"] is not None:
                            zx, zy = state["latest_z"]
                            sc1.set_offsets([[zx, zy]])
                            radius_line.set_data([0.0, zx], [0.0, zy])
                            ang = float(np.arctan2(zy, zx))
                            angle_text.set_text(
                                f"Zero pose for {name}\n"
                                f"angle: {ang:+.3f} rad ({np.degrees(ang):+.1f}°)\n"
                                f"samples: {len(zero_samples)}/{SAMPLE_COUNT}\n"
                                f"hold still, press 'n'"
                            )
                        else:
                            sc1.set_offsets(np.empty((0, 2)))
                            radius_line.set_data([], [])
                            angle_text.set_text("(waiting for data)")
                        ax1.draw_artist(sc1)

                    ax1.draw_artist(radius_line)
                    ax1.draw_artist(angle_text)

                    fig.canvas.blit(ax0.bbox)
                    fig.canvas.blit(ax1.bbox)
                    fig.canvas.flush_events()
                    last_draw = t

                plt.pause(0.001)

        finally:
            plt.close(fig)


# =============================================================================
# IK HELPER
# =============================================================================

def _get_frame_id(model, name: str) -> int | None:
    """Get frame ID if it exists, else None."""
    try:
        fid = model.getFrameId(name)
        if 0 <= fid < model.nframes:
            return fid
    except Exception:
        pass
    return None


class ExoskeletonIKHelper:
    """
    Helper class for IK-based teleoperation.
    
    Loads exoskeleton URDFs and computes FK to get end-effector poses,
    then uses G1 IK to solve for joint angles.
    
    Exoskeleton URDFs are automatically loaded from lerobot/unitree-g1-mujoco:
    - assets/exo_left.urdf with assets/meshes_exo_left/
    - assets/exo_right.urdf with assets/meshes_exo_right/
    
    Note: Creates symlinks assets_left -> meshes_exo_left to resolve URDF package:// paths.
    """

    def __init__(
        self,
        frozen_joints: list[str] | None = None,
    ):
        try:
            import pinocchio as pin
        except ImportError as e:
            raise ImportError(
                "IK mode requires pinocchio. Install with: pip install pin"
            ) from e

        self.pin = pin
        self.ee_frame = "ee"
        self.frozen_joints = frozen_joints or []
        
        # Download model repo (same as G1 IK)
        repo_path = snapshot_download("lerobot/unitree-g1-mujoco")
        assets_dir = os.path.join(repo_path, "assets")
        
        # Create symlinks for URDF package:// paths
        # URDFs use package://assets_left/ but actual folders are meshes_exo_left/
        for pkg_name, mesh_folder in [("assets_left", "meshes_exo_left"), ("assets_right", "meshes_exo_right")]:
            symlink_path = os.path.join(assets_dir, pkg_name)
            target_path = os.path.join(assets_dir, mesh_folder)
            if not os.path.exists(symlink_path) and os.path.exists(target_path):
                try:
                    os.symlink(target_path, symlink_path)
                    logger.info(f"Created symlink: {symlink_path} -> {target_path}")
                except OSError as e:
                    logger.warning(f"Could not create symlink {symlink_path}: {e}")
        
        # Exoskeleton URDF paths (mesh_dir is assets/ so Pinocchio finds package://assets_left/)
        left_urdf = os.path.join(assets_dir, "exo_left.urdf")
        left_mesh_dir = assets_dir
        right_urdf = os.path.join(assets_dir, "exo_right.urdf")
        right_mesh_dir = assets_dir
        
        # Load G1 IK solver
        from lerobot.robots.unitree_g1.robot_kinematic_processor import G1_29_ArmIK
        self.g1_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
        self.robot_g1 = self.g1_ik.reduced_robot
        self.robot_g1.data = self.robot_g1.model.createData()
        self.q_g1 = pin.neutral(self.robot_g1.model)
        
        # Build frozen joint index map
        self.frozen_joint_indices = {}
        for jname in self.frozen_joints:
            if jname in self.robot_g1.model.names:
                jid = self.robot_g1.model.getJointId(jname)
                idx_q = self.robot_g1.model.idx_qs[jid]
                self.frozen_joint_indices[jname] = idx_q
                logger.info(f"Freezing joint: {jname} (q_idx={idx_q})")

        # G1 end-effector frame IDs
        self.left_ee_id = _get_frame_id(self.robot_g1.model, "L_ee")
        self.right_ee_id = _get_frame_id(self.robot_g1.model, "R_ee")
        
        # Load exoskeleton models
        self.exo_left = None
        self.exo_right = None
        self.exo_left_ee_id = None
        self.exo_right_ee_id = None
        self.q_exo_left = None
        self.q_exo_right = None
        
        if os.path.exists(left_urdf):
            self.exo_left = pin.RobotWrapper.BuildFromURDF(left_urdf, left_mesh_dir)
            self.q_exo_left = pin.neutral(self.exo_left.model)
            self.exo_left_ee_id = self._find_ee_frame(self.exo_left.model)
            logger.info(f"Loaded left exo URDF: {left_urdf}, EE frame: {self.exo_left.model.frames[self.exo_left_ee_id].name}")
        else:
            logger.warning(f"Left exo URDF not found: {left_urdf}")
        
        if os.path.exists(right_urdf):
            self.exo_right = pin.RobotWrapper.BuildFromURDF(right_urdf, right_mesh_dir)
            self.q_exo_right = pin.neutral(self.exo_right.model)
            self.exo_right_ee_id = self._find_ee_frame(self.exo_right.model)
            logger.info(f"Loaded right exo URDF: {right_urdf}, EE frame: {self.exo_right.model.frames[self.exo_right_ee_id].name}")
        else:
            logger.warning(f"Right exo URDF not found: {right_urdf}")

        # Joint maps for exo models
        self.exo_left_joint_map = self._build_joint_map(self.exo_left) if self.exo_left else {}
        self.exo_right_joint_map = self._build_joint_map(self.exo_right) if self.exo_right else {}
        
        # Visualization (initialized lazily)
        self.viz_g1 = None
        self.viz_exo_left = None
        self.viz_exo_right = None
        self.viewer = None
        self.left_offset = np.array([0.6, 0.3, 0.0])
        self.right_offset = np.array([0.6, -0.3, 0.0])

    def init_visualization(self, show_axes: bool = True):
        """Initialize Meshcat visualization for G1 and exoskeletons."""
        try:
            import meshcat.geometry as mg
            from pinocchio.visualize import MeshcatVisualizer
        except ImportError as e:
            logger.warning(f"Meshcat visualization not available: {e}")
            return
        
        pin = self.pin
        
        # Initialize G1 visualization
        self.viz_g1 = MeshcatVisualizer(
            self.robot_g1.model, self.robot_g1.collision_model, self.robot_g1.visual_model
        )
        self.viz_g1.initViewer(open=True)
        self.viz_g1.loadViewerModel("g1")
        self.viz_g1.display(self.q_g1)
        self.viewer = self.viz_g1.viewer
        
        # Initialize left exo visualization
        if self.exo_left is not None:
            self.viz_exo_left = MeshcatVisualizer(
                self.exo_left.model, self.exo_left.collision_model, self.exo_left.visual_model
            )
            self.viz_exo_left.initViewer(open=False)
            self.viz_exo_left.viewer = self.viewer
            self.viz_exo_left.loadViewerModel("exo_left")
            # Set display offset
            T = np.eye(4)
            T[:3, 3] = self.left_offset
            self.viewer["exo_left"].set_transform(T)
            self.viz_exo_left.display(self.q_exo_left)
        
        # Initialize right exo visualization
        if self.exo_right is not None:
            self.viz_exo_right = MeshcatVisualizer(
                self.exo_right.model, self.exo_right.collision_model, self.exo_right.visual_model
            )
            self.viz_exo_right.initViewer(open=False)
            self.viz_exo_right.viewer = self.viewer
            self.viz_exo_right.loadViewerModel("exo_right")
            # Set display offset
            T = np.eye(4)
            T[:3, 3] = self.right_offset
            self.viewer["exo_right"].set_transform(T)
            self.viz_exo_right.display(self.q_exo_right)
        
        # Add marker spheres
        self._add_sphere("markers/left_exo_ee", 0.012, (0.2, 1.0, 0.2, 0.9))
        self._add_sphere("markers/left_g1_ee", 0.015, (1.0, 0.2, 0.2, 0.9))
        self._add_sphere("markers/left_ik_target", 0.015, (0.1, 0.3, 1.0, 0.9))
        self._add_sphere("markers/right_exo_ee", 0.012, (0.2, 1.0, 0.2, 0.9))
        self._add_sphere("markers/right_g1_ee", 0.015, (1.0, 0.2, 0.2, 0.9))
        self._add_sphere("markers/right_ik_target", 0.015, (0.1, 0.3, 1.0, 0.9))
        
        if show_axes:
            self._add_axes("markers/left_exo_axes", 0.06)
            self._add_axes("markers/left_g1_axes", 0.08)
            self._add_axes("markers/right_exo_axes", 0.06)
            self._add_axes("markers/right_g1_axes", 0.08)
        
        logger.info(f"Meshcat visualization initialized: {self.viewer.url()}")
        print(f"\n🌐 Meshcat URL: {self.viewer.url()}\n")
    
    def _add_sphere(self, path: str, radius: float, rgba: tuple):
        """Add a colored sphere marker."""
        if self.viewer is None:
            return
        import meshcat.geometry as mg
        color_int = int(rgba[0] * 255) << 16 | int(rgba[1] * 255) << 8 | int(rgba[2] * 255)
        self.viewer[path].set_object(
            mg.Sphere(radius),
            mg.MeshPhongMaterial(color=color_int, opacity=rgba[3], transparent=rgba[3] < 1.0),
        )
    
    def _add_axes(self, path: str, axis_len: float = 0.1, axis_w: int = 6):
        """Add XYZ axes visualization."""
        if self.viewer is None:
            return
        import meshcat.geometry as mg
        pts = np.array([
            [0, 0, 0], [axis_len, 0, 0],
            [0, 0, 0], [0, axis_len, 0],
            [0, 0, 0], [0, 0, axis_len],
        ], dtype=np.float32).T
        cols = np.array([
            [1, 0, 0], [1, 0, 0],
            [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1],
        ], dtype=np.float32).T
        self.viewer[path].set_object(
            mg.LineSegments(
                mg.PointsGeometry(position=pts, color=cols),
                mg.LineBasicMaterial(linewidth=axis_w, vertexColors=True),
            )
        )
    
    def update_visualization(self):
        """Update Meshcat display with current joint states."""
        if self.viewer is None:
            return
        
        pin = self.pin
        
        # Update G1 display
        if self.viz_g1 is not None:
            self.viz_g1.display(self.q_g1)
            
            # Update G1 EE markers
            pin.forwardKinematics(self.robot_g1.model, self.robot_g1.data, self.q_g1)
            pin.updateFramePlacements(self.robot_g1.model, self.robot_g1.data)
            
            if self.left_ee_id is not None:
                T = self.robot_g1.data.oMf[self.left_ee_id].homogeneous
                self.viewer["markers/left_g1_ee"].set_transform(T)
                self.viewer["markers/left_g1_axes"].set_transform(T)
            
            if self.right_ee_id is not None:
                T = self.robot_g1.data.oMf[self.right_ee_id].homogeneous
                self.viewer["markers/right_g1_ee"].set_transform(T)
                self.viewer["markers/right_g1_axes"].set_transform(T)
        
        # Update left exo display
        if self.viz_exo_left is not None and self.exo_left is not None:
            self.viz_exo_left.display(self.q_exo_left)
            
            pin.forwardKinematics(self.exo_left.model, self.exo_left.data, self.q_exo_left)
            pin.updateFramePlacements(self.exo_left.model, self.exo_left.data)
            T_exo_ee = self.exo_left.data.oMf[self.exo_left_ee_id]
            
            # EE marker in world coords (with offset)
            T_world = pin.SE3(np.eye(3), self.left_offset) * T_exo_ee
            self.viewer["markers/left_exo_ee"].set_transform(T_world.homogeneous)
            self.viewer["markers/left_exo_axes"].set_transform(T_world.homogeneous)
            
            # IK target marker
            target_pos = self.left_offset + T_exo_ee.translation
            T_target = np.eye(4)
            T_target[:3, :3] = T_exo_ee.rotation
            T_target[:3, 3] = target_pos
            self.viewer["markers/left_ik_target"].set_transform(T_target)
        
        # Update right exo display
        if self.viz_exo_right is not None and self.exo_right is not None:
            self.viz_exo_right.display(self.q_exo_right)
            
            pin.forwardKinematics(self.exo_right.model, self.exo_right.data, self.q_exo_right)
            pin.updateFramePlacements(self.exo_right.model, self.exo_right.data)
            T_exo_ee = self.exo_right.data.oMf[self.exo_right_ee_id]
            
            # EE marker in world coords (with offset)
            T_world = pin.SE3(np.eye(3), self.right_offset) * T_exo_ee
            self.viewer["markers/right_exo_ee"].set_transform(T_world.homogeneous)
            self.viewer["markers/right_exo_axes"].set_transform(T_world.homogeneous)
            
            # IK target marker
            target_pos = self.right_offset + T_exo_ee.translation
            T_target = np.eye(4)
            T_target[:3, :3] = T_exo_ee.rotation
            T_target[:3, 3] = target_pos
            self.viewer["markers/right_ik_target"].set_transform(T_target)

    def _find_ee_frame(self, model) -> int:
        """Find end-effector frame in model."""
        ee_id = _get_frame_id(model, self.ee_frame)
        if ee_id is not None:
            return ee_id
        # Fallback: find last body frame
        for fid in reversed(range(model.nframes)):
            if model.frames[fid].type == self.pin.FrameType.BODY:
                return fid
        return 0

    def _build_joint_map(self, robot) -> dict[str, int]:
        """Build mapping from joint names to q indices."""
        joint_map = {}
        for jname in [j[0] for j in JOINTS]:  # exo joint names
            if jname in robot.model.names:
                jid = robot.model.getJointId(jname)
                joint_map[jname] = robot.model.idx_qs[jid]
        return joint_map

    def compute_g1_joints_from_exo(
        self,
        left_angles: dict[str, float],
        right_angles: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute G1 joint angles from exoskeleton joint angles using IK.
        
        1. Set exo joint angles
        2. Compute exo FK to get end-effector poses
        3. Run G1 IK to solve for joint angles
        """
        pin = self.pin
        
        # Update left exo FK
        left_target = None
        if self.exo_left is not None and left_angles:
            for name, angle in left_angles.items():
                if name in self.exo_left_joint_map:
                    self.q_exo_left[self.exo_left_joint_map[name]] = float(angle)
            
            pin.forwardKinematics(self.exo_left.model, self.exo_left.data, self.q_exo_left)
            pin.updateFramePlacements(self.exo_left.model, self.exo_left.data)
            T_exo_ee = self.exo_left.data.oMf[self.exo_left_ee_id]
            # Add display offset to IK target (maps exo workspace to G1 workspace)
            target_pos = self.left_offset + T_exo_ee.translation
            left_target = np.eye(4)
            left_target[:3, :3] = T_exo_ee.rotation
            left_target[:3, 3] = target_pos
        
        # Update right exo FK
        right_target = None
        if self.exo_right is not None and right_angles:
            for name, angle in right_angles.items():
                if name in self.exo_right_joint_map:
                    self.q_exo_right[self.exo_right_joint_map[name]] = float(angle)
            
            pin.forwardKinematics(self.exo_right.model, self.exo_right.data, self.q_exo_right)
            pin.updateFramePlacements(self.exo_right.model, self.exo_right.data)
            T_exo_ee = self.exo_right.data.oMf[self.exo_right_ee_id]
            # Add display offset to IK target (maps exo workspace to G1 workspace)
            target_pos = self.right_offset + T_exo_ee.translation
            right_target = np.eye(4)
            right_target[:3, :3] = T_exo_ee.rotation
            right_target[:3, 3] = target_pos
        
        # Get current G1 poses if targets not available
        pin.forwardKinematics(self.robot_g1.model, self.robot_g1.data, self.q_g1)
        pin.updateFramePlacements(self.robot_g1.model, self.robot_g1.data)
        
        if left_target is None and self.left_ee_id is not None:
            left_target = self.robot_g1.data.oMf[self.left_ee_id].homogeneous
        if right_target is None and self.right_ee_id is not None:
            right_target = self.robot_g1.data.oMf[self.right_ee_id].homogeneous
        
        if left_target is None or right_target is None:
            logger.warning("Missing IK targets, returning current pose")
            return {}
        
        # Save frozen joint values
        frozen_values = {name: self.q_g1[idx] for name, idx in self.frozen_joint_indices.items()}
        
        # Solve IK
        self.q_g1, _ = self.g1_ik.solve_ik(left_target, right_target, current_lr_arm_motor_q=self.q_g1)
        
        # Restore frozen values
        for name, idx in self.frozen_joint_indices.items():
            self.q_g1[idx] = frozen_values[name]
        
        # Convert q_g1 to action dict
        action_dict = {}
        for i, joint in enumerate(G1_29_JointArmIndex):
            if i < len(self.q_g1):
                action_dict[f"{joint.name}.q"] = float(self.q_g1[i])
        
        return action_dict


# =============================================================================
# UNITREE G1 TELEOPERATOR
# =============================================================================

class UnitreeG1Teleoperator(Teleoperator):
    """
    Bimanual exoskeleton arms teleoperator for Unitree G1 arms.
    
    Uses inverse kinematics: exoskeleton FK computes end-effector pose,
    G1 IK solves for joint angles.
    """

    config_class = UnitreeG1TeleoperatorConfig
    name = "unitree_g1"

    def __init__(self, config: UnitreeG1TeleoperatorConfig):
        super().__init__(config)
        self.config = config

        # Resolve serial ports
        left_port, right_port = self._resolve_ports(
            config.left_arm_config.port, config.right_arm_config.port
        )

        # Setup calibration directory and file paths
        self.calibration_dir = (
            config.calibration_dir
            if config.calibration_dir
            else HF_LEROBOT_CALIBRATION / TELEOPERATORS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

        left_id = f"{config.id}_left" if config.id else "left"
        right_id = f"{config.id}_right" if config.id else "right"
        
        left_calib_fpath = self.calibration_dir / f"{left_id}.json"
        right_calib_fpath = self.calibration_dir / f"{right_id}.json"

        # Create exoskeleton arm instances
        self.left_arm = ExoskeletonArm(
            port=left_port,
            baud_rate=config.left_arm_config.baud_rate,
            calibration_fpath=left_calib_fpath,
            side="left",
        )
        self.right_arm = ExoskeletonArm(
            port=right_port,
            baud_rate=config.right_arm_config.baud_rate,
            calibration_fpath=right_calib_fpath,
            side="right",
        )
        
        # IK helper (initialized on connect)
        self.ik_helper: ExoskeletonIKHelper | None = None

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.q": float for name in self._g1_joint_names}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        
        # Initialize IK helper (auto-loads exo URDFs from lerobot/unitree-g1-mujoco)
        frozen_joints = [j.strip() for j in self.config.frozen_joints.split(",") if j.strip()]
        self.ik_helper = ExoskeletonIKHelper(frozen_joints=frozen_joints)
        logger.info("IK helper initialized")
        
        # Initialize Meshcat visualization if enabled
        if self.config.visualize:
            self.ik_helper.init_visualization(show_axes=self.config.show_axes)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        """Run interactive calibration for both exoskeleton arms.
        
        If calibration already exists for both arms, this is a no-op.
        """
        left_calibrated = self.left_arm.is_calibrated
        right_calibrated = self.right_arm.is_calibrated
        
        if left_calibrated and right_calibrated:
            logger.info("Calibration already exists for both arms. Skipping interactive calibration.")
            return
        
        if not left_calibrated:
            logger.info("Starting calibration for left arm...")
            self.left_arm.calibrate()
        else:
            logger.info("Left arm already calibrated. Skipping.")
        
        if not right_calibrated:
            logger.info("Starting calibration for right arm...")
            self.right_arm.calibrate()
        else:
            logger.info("Right arm already calibrated. Skipping.")

    def configure(self) -> None:
        """No additional configuration needed for exoskeleton arms."""
        pass

    def get_action(self) -> dict[str, float]:
        """Get current joint angles mapped to G1 action format via IK."""
        left_angles = self.left_arm.get_angles()
        right_angles = self.right_arm.get_angles()

        # Default all joints to 0.0 so dataset logging has full action keys
        action_dict: dict[str, float] = {f"{name}.q": 0.0 for name in self._g1_joint_names}
        
        if self.ik_helper is not None:
            ik_action = self.ik_helper.compute_g1_joints_from_exo(left_angles, right_angles)
            action_dict.update(ik_action)
            
            # Update Meshcat visualization if enabled
            if self.config.visualize:
                self.ik_helper.update_visualization()
        
        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("Exoskeleton arms do not support feedback")

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()

    def run_visualization_loop(self):
        """
        Run interactive Meshcat visualization loop.
        
        This is typically called after calibration to verify exoskeleton tracking.
        Press Ctrl+C to exit.
        """
        # Initialize IK helper if not already done
        if self.ik_helper is None:
            frozen_joints = [j.strip() for j in self.config.frozen_joints.split(",") if j.strip()]
            self.ik_helper = ExoskeletonIKHelper(frozen_joints=frozen_joints)
        
        # Initialize visualization
        self.ik_helper.init_visualization(show_axes=self.config.show_axes)
        
        print("\n" + "=" * 60)
        print("Visualization running! Move the exoskeletons to test tracking.")
        print("Press Ctrl+C to exit.")
        print("=" * 60 + "\n")
        
        last_print = 0.0
        try:
            while True:
                # Read exoskeleton angles
                left_angles = self.left_arm.get_angles()
                right_angles = self.right_arm.get_angles()
                
                # Compute IK and update visualization
                if self.ik_helper is not None:
                    self.ik_helper.compute_g1_joints_from_exo(left_angles, right_angles)
                    self.ik_helper.update_visualization()
                
                # Print status periodically
                now = time.time()
                if now - last_print > 0.5:
                    status = []
                    if self.ik_helper.exo_left is not None:
                        T_ee = self.ik_helper.exo_left.data.oMf[self.ik_helper.exo_left_ee_id]
                        pos = T_ee.translation
                        status.append(f"L: [{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]")
                    if self.ik_helper.exo_right is not None:
                        T_ee = self.ik_helper.exo_right.data.oMf[self.ik_helper.exo_right_ee_id]
                        pos = T_ee.translation
                        status.append(f"R: [{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]")
                    if status:
                        print(f"EE pos: {' | '.join(status)}", end="\r")
                    last_print = now
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\nVisualization stopped.")

    @cached_property
    def _g1_joint_names(self) -> list[str]:
        return [joint.name for joint in G1_29_JointIndex]

    def _resolve_ports(self, left_port: str, right_port: str) -> tuple[str, str]:
        """Pick /dev/ttyACM{0,1} if either requested port is missing or empty."""
        available = []
        for candidate in ("/dev/ttyACM0", "/dev/ttyACM1"):
            if os.path.exists(candidate):
                available.append(candidate)

        def _is_valid(port: str) -> bool:
            return bool(port) and os.path.exists(port)

        if _is_valid(left_port) and _is_valid(right_port) and left_port != right_port:
            return left_port, right_port

        if len(available) >= 2:
            return available[0], available[1]

        # Fallback to whatever was provided; ExoskeletonArm will surface errors
        return left_port, right_port
