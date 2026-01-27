"""Exoskeleton calibration: dataclasses, transform helpers, and interactive calibration."""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import serial

logger = logging.getLogger(__name__)

# Constants
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


@dataclass(frozen=True)
class CalibParams:
    """Calibration parameters."""

    fit_every: float = 0.15
    min_fit_points: int = 60
    fit_window: int = 900
    max_fit_points: int = 300
    trim_low: float = 0.05
    trim_high: float = 0.95
    median_window: int = 5
    history: int = 3500
    draw_hz: float = 120.0
    sample_count: int = 50


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def joint_z_and_angle(raw16: list[int], j: ExoskeletonJointCalibration) -> tuple[np.ndarray, float]:
    """
    Compute transformed coords (z) and angle from raw sensor reading.

    This is the core transform: raw → centered → ellipse-to-circle → angle.
    """
    s, c = raw16[j.pair[0]], raw16[j.pair[1]]
    if j.flipped:
        s, c = ADC_MAX - s, ADC_MAX - c

    p = np.array([float(c) - j.center_guess, float(s) - j.center_guess])
    z = np.asarray(j.T) @ (p - np.asarray(j.center_fit))
    ang = float(np.arctan2(z[1], z[0])) - j.zero_offset
    return z, normalize_angle(-ang)


def exo_raw_to_angles(raw16: list[int], calib: ExoskeletonCalibration) -> dict[str, float]:
    """Convert raw sensor readings to joint angles using calibration."""
    return {j.name: joint_z_and_angle(raw16, j)[1] for j in calib.joints}


def run_exo_calibration(
    ser: serial.Serial,
    side: str,
    save_path: Path | None = None,
    params: CalibParams | None = None,
) -> ExoskeletonCalibration:
    """
    Run interactive calibration for an exoskeleton arm.

    Opens a matplotlib window for ellipse fitting and zero-pose capture.
    Returns the completed calibration.
    """
    try:
        import cv2
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Calibration requires matplotlib and opencv-python. "
            "Install with: pip install matplotlib opencv-python"
        ) from e

    from .exo_serial import parse_raw16

    params = params or CalibParams()
    logger.info(f"Starting calibration for {side} exoskeleton arm")

    def running_median(win: deque) -> float:
        return float(np.median(np.fromiter(win, dtype=float)))

    def read_joint_point(raw16: list[int], pair: tuple[int, int], flipped: bool, center: float):
        s, c = raw16[pair[0]], raw16[pair[1]]
        if flipped:
            s, c = ADC_MAX - s, ADC_MAX - c
        return float(c) - center, float(s) - center, float(s), float(c)

    def select_fit_subset(xs, ys):
        n = min(params.fit_window, len(xs))
        if n <= 0:
            return None, None
        x = np.asarray(list(xs)[-n:], dtype=float)
        y = np.asarray(list(ys)[-n:], dtype=float)
        r = np.sqrt(x * x + y * y)
        if len(r) >= 20:
            lo, hi = np.quantile(r, params.trim_low), np.quantile(r, params.trim_high)
            keep = (r >= lo) & (r <= hi)
            x, y = x[keep], y[keep]
        if len(x) > params.max_fit_points:
            idx = np.linspace(0, len(x) - 1, params.max_fit_points).astype(int)
            x, y = x[idx], y[idx]
        return x, y

    def fit_ellipse_opencv(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if len(x) < 5:
            return None
        pts = np.stack([x, y], axis=1).astype(np.float32).reshape(-1, 1, 2)
        try:
            (xc, yc), (w, h), angle_deg = cv2.fitEllipse(pts)
        except cv2.error:
            return None
        a, b = float(w) * 0.5, float(h) * 0.5
        phi = np.deg2rad(float(angle_deg))
        if b > a:
            a, b = b, a
            phi += np.pi / 2.0
        if not np.isfinite(a) or not np.isfinite(b) or a <= 1e-6 or b <= 1e-6:
            return None
        cp, sp = float(np.cos(phi)), float(np.sin(phi))
        rot = np.array([[cp, -sp], [sp, cp]], dtype=float)
        center = np.array([float(xc), float(yc)], dtype=float)
        tt = np.linspace(0, 2 * np.pi, 360)
        outline = (rot @ np.stack([a * np.cos(tt), b * np.sin(tt)])).T + center
        return {"center": center, "a": a, "b": b, "R": rot, "ex": outline[:, 0], "ey": outline[:, 1]}

    def read_raw_from_serial() -> list[int] | None:
        last = None
        while ser.in_waiting > 0:
            b = ser.readline()
            if not b:
                break
            raw16 = parse_raw16(b.decode("utf-8", errors="ignore"))
            if raw16 is not None:
                last = raw16
        if last is None:
            b = ser.readline()
            if b:
                last = parse_raw16(b.decode("utf-8", errors="ignore"))
        return last

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
    angle_text = ax1.text(
        0.02, 0.98, "", transform=ax1.transAxes, va="top", ha="left", fontsize=12, animated=True
    )

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

    def reset_state():
        return {
            "xs": deque(maxlen=params.history),
            "ys": deque(maxlen=params.history),
            "xu": deque(maxlen=params.history),
            "yu": deque(maxlen=params.history),
            "win_s": deque(maxlen=params.median_window),
            "win_c": deque(maxlen=params.median_window),
            "ellipse_cache": None,
            "T": None,
            "center_fit": None,
            "have_transform": False,
            "latest_z": None,
            "last_fit": 0.0,
        }

    state = reset_state()
    last_draw = 0.0
    name, pair, flipped = JOINTS[joint_idx]
    fig.canvas.manager.set_window_title(f"[{joint_idx + 1}/{len(JOINTS)}] {name} - ELLIPSE")
    ax0.set_title(f"{name} raw (filtered)")
    logger.info(f"[{joint_idx + 1}/{len(JOINTS)}] Calibrating {name}")
    logger.info("Step 1: Move joint around to map ellipse, then press 'n'")

    try:
        while plt.fignum_exists(fig.number):
            name, pair, flipped = JOINTS[joint_idx]

            # State machine: ellipse → zero_pose → next joint
            if phase == "ellipse" and advance_requested and state["have_transform"]:
                joints_out.append(
                    {
                        "name": name,
                        "pair": [int(pair[0]), int(pair[1])],
                        "flipped": bool(flipped),
                        "center_guess": float(CENTER_GUESS),
                        "center_fit": state["center_fit"].tolist(),
                        "T": state["T"].tolist(),
                    }
                )
                logger.info(f"  -> Ellipse saved for {name}")
                phase, zero_samples, advance_requested = "zero_pose", [], False
                fig.canvas.manager.set_window_title(f"[{joint_idx + 1}/{len(JOINTS)}] {name} - ZERO POSE")
                ax0.set_title(f"{name} - hold zero pose")
                fig.canvas.draw()
                bg0, bg1 = fig.canvas.copy_from_bbox(ax0.bbox), fig.canvas.copy_from_bbox(ax1.bbox)
                logger.info(f"Step 2: Hold {name} in zero position, then press 'n'")

            elif phase == "ellipse" and advance_requested and not state["have_transform"]:
                logger.info("  (Need valid fit first - keep moving the joint)")
                advance_requested = False

            elif phase == "zero_pose" and advance_requested:
                if len(zero_samples) >= params.sample_count:
                    zero_offset = float(np.mean(zero_samples[-params.sample_count :]))
                    joints_out[-1]["zero_offset"] = zero_offset
                    logger.info(f"  -> {name} zero: {zero_offset:+.3f} rad ({np.degrees(zero_offset):+.1f}°)")
                    joint_idx += 1
                    advance_requested = False

                    if joint_idx >= len(JOINTS):
                        # All joints done
                        calib = ExoskeletonCalibration(
                            version=2,
                            side=side,
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
                        if save_path:
                            save_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(save_path, "w") as f:
                                json.dump(calib.to_dict(), f, indent=2)
                            logger.info(f"Saved calibration to {save_path}")
                        logger.info("Calibration complete!")
                        plt.close(fig)
                        return calib

                    # Next joint
                    phase, state = "ellipse", reset_state()
                    name, pair, flipped = JOINTS[joint_idx]
                    fig.canvas.manager.set_window_title(f"[{joint_idx + 1}/{len(JOINTS)}] {name} - ELLIPSE")
                    ax0.set_title(f"{name} raw (filtered)")
                    fig.canvas.draw()
                    bg0, bg1 = fig.canvas.copy_from_bbox(ax0.bbox), fig.canvas.copy_from_bbox(ax1.bbox)
                    logger.info(f"[{joint_idx + 1}/{len(JOINTS)}] Calibrating {name}")
                    logger.info("Step 1: Move joint around to map ellipse, then press 'n'")
                else:
                    logger.info(
                        f"  (Collecting samples: {len(zero_samples)}/{params.sample_count} - hold still)"
                    )
                    advance_requested = False

            # Read sensor
            raw16 = read_raw_from_serial()
            if raw16 is not None:
                x_raw, y_raw, s_raw, c_raw = read_joint_point(raw16, pair, flipped, CENTER_GUESS)

                if phase == "ellipse":
                    if state["have_transform"]:
                        z = state["T"] @ (np.array([x_raw, y_raw]) - state["center_fit"])
                        state["xu"].append(float(z[0]))
                        state["yu"].append(float(z[1]))
                        state["latest_z"] = (float(z[0]), float(z[1]))
                    state["win_s"].append(s_raw)
                    state["win_c"].append(c_raw)
                    if len(state["win_s"]) >= max(3, params.median_window):
                        state["ys"].append(running_median(state["win_s"]) - CENTER_GUESS)
                        state["xs"].append(running_median(state["win_c"]) - CENTER_GUESS)
                else:
                    jdata = joints_out[-1]
                    z = np.array(jdata["T"]) @ (np.array([x_raw, y_raw]) - np.array(jdata["center_fit"]))
                    zero_samples.append(float(np.arctan2(z[1], z[0])))
                    state["latest_z"] = (float(z[0]), float(z[1]))

            # Ellipse fitting
            t = time.time()
            if (
                phase == "ellipse"
                and (t - state["last_fit"]) >= params.fit_every
                and len(state["xs"]) >= params.min_fit_points
            ):
                xfit, yfit = select_fit_subset(state["xs"], state["ys"])
                if xfit is not None and len(xfit) >= params.min_fit_points:
                    fit = fit_ellipse_opencv(xfit, yfit)
                    if fit is not None:
                        state["center_fit"] = fit["center"]
                        state["T"] = np.diag([1.0 / fit["a"], 1.0 / fit["b"]]) @ fit["R"].T
                        state["ellipse_cache"] = (fit["ex"], fit["ey"])
                        state["have_transform"] = True
                state["last_fit"] = t

            # Drawing
            if (t - last_draw) >= 1.0 / params.draw_hz:
                fig.canvas.restore_region(bg0)
                fig.canvas.restore_region(bg1)

                if phase == "ellipse":
                    sc0.set_offsets(np.c_[state["xs"], state["ys"]] if state["xs"] else np.empty((0, 2)))
                    ax0.draw_artist(sc0)
                    ell_line.set_data(*state["ellipse_cache"] if state["ellipse_cache"] else ([], []))
                    ax0.draw_artist(ell_line)
                    sc1.set_offsets(np.c_[state["xu"], state["yu"]] if state["xu"] else np.empty((0, 2)))
                    ax1.draw_artist(sc1)
                    if state["latest_z"]:
                        zx, zy = state["latest_z"]
                        radius_line.set_data([0.0, zx], [0.0, zy])
                        ang = float(np.arctan2(zy, zx))
                        angle_text.set_text(
                            f"angle: {ang:+.3f} rad  ({np.degrees(ang):+.1f}°)\nmove {name}, press 'n' to advance"
                        )
                    else:
                        radius_line.set_data([], [])
                        angle_text.set_text("(waiting for fit)")
                else:
                    sc0.set_offsets(np.empty((0, 2)))
                    ax0.draw_artist(sc0)
                    ell_line.set_data([], [])
                    ax0.draw_artist(ell_line)
                    if state["latest_z"]:
                        zx, zy = state["latest_z"]
                        sc1.set_offsets([[zx, zy]])
                        radius_line.set_data([0.0, zx], [0.0, zy])
                        ang = float(np.arctan2(zy, zx))
                        angle_text.set_text(
                            f"Zero pose for {name}\nangle: {ang:+.3f} rad\nsamples: {len(zero_samples)}/{params.sample_count}\nhold still, press 'n'"
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

    # If we get here, window was closed early
    raise RuntimeError("Calibration window closed before completion")
