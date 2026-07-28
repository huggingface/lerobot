from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

CALIBRATION_BOUNDS_DEG = np.asarray(
    [
        [-109.49450549450549, 109.49450549450549],
        [-98.94505494505495, 98.94505494505495],
        [-95.91208791208791, 95.91208791208791],
        [-94.76923076923077, 94.76923076923077],
        [-167.64835164835165, 167.64835164835165],
        [0.0, 100.0],
    ],
    dtype=np.float64,
)

EXPECTED_CALIBRATION_SHA256 = "c78e4f7e1383571c6aa496f62996f518b3e4122f78244d2bbc094658bc0cb8a0"
MAX_RELATIVE_TARGET = 5.0
SIM_STATE_CALIBRATION_TOLERANCE = 0.01


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_calibration(path: str | Path) -> str:
    path = Path(path)
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError(
            "Follower calibration changed since the Task1 safety bounds were frozen; "
            "refuse hardware execution until bounds are regenerated and reviewed."
        )

    payload = json.loads(path.read_text())
    if tuple(payload) != JOINT_ORDER:
        raise RuntimeError("Follower calibration joint order differs from the frozen Task1 contract.")
    for joint in JOINT_ORDER:
        record = payload[joint]
        if not isinstance(record.get("range_min"), int) or not isinstance(record.get("range_max"), int):
            raise RuntimeError(f"Follower calibration range is malformed for {joint}.")
        if record["range_min"] >= record["range_max"]:
            raise RuntimeError(f"Follower calibration range is not increasing for {joint}.")
    return actual_hash


def clamp_action_fail_closed(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (6,):
        raise RuntimeError(f"Policy action must have shape (6,), got {values.shape}.")
    if not np.isfinite(values).all():
        raise RuntimeError("Policy action contains NaN or infinity; refusing to send any action.")

    lower = CALIBRATION_BOUNDS_DEG[:, 0]
    upper = CALIBRATION_BOUNDS_DEG[:, 1]
    clipped = np.clip(values, lower, upper)
    clip_mask = clipped != values
    if not np.isfinite(clipped).all():
        raise RuntimeError("Calibration clipping produced a non-finite action.")
    return clipped.astype(np.float32), clip_mask


def project_sim_state_to_calibration(
    state: np.ndarray,
    tolerance: float = SIM_STATE_CALIBRATION_TOLERANCE,
) -> dict[str, np.ndarray]:
    """Project only negligible simulator boundary noise into real calibration.

    Nexus intentionally perturbs reset qpos and settles the model. Its dataset
    conversion can therefore report values a few thousandths below the gripper
    lower bound. This function is only for the hardware-free simulator input
    seam. The real deployment path remains strict and output calibration
    clipping is unchanged.
    """
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (6,):
        raise RuntimeError(
            f"Simulator observation state must have shape (6,), got {values.shape}."
        )
    if not np.isfinite(values).all():
        raise RuntimeError(
            "Simulator observation state contains NaN or infinity."
        )
    if not np.isfinite(tolerance) or tolerance < 0:
        raise RuntimeError(
            "Simulator calibration tolerance must be a finite non-negative scalar."
        )

    lower = CALIBRATION_BOUNDS_DEG[:, 0]
    upper = CALIBRATION_BOUNDS_DEG[:, 1]
    below_by = np.maximum(lower - values, 0.0)
    above_by = np.maximum(values - upper, 0.0)
    if ((below_by > tolerance) | (above_by > tolerance)).any():
        raise RuntimeError(
            "Simulator observation state exceeds the bounded calibration "
            "projection tolerance; possible state-coordinate contract mismatch."
        )
    projected = np.clip(values, lower, upper)
    return {
        "state": projected.astype(np.float32),
        "projection_mask": projected != values,
        "projection_delta": (projected - values).astype(np.float32),
    }


def apply_action_safety(
    raw_action: np.ndarray,
    current_state: np.ndarray,
    max_relative_target: float = MAX_RELATIVE_TARGET,
) -> dict[str, np.ndarray]:
    state = np.asarray(current_state, dtype=np.float64)
    if state.shape != (6,):
        raise RuntimeError(f"Observation state must have shape (6,), got {state.shape}.")
    if not np.isfinite(state).all():
        raise RuntimeError("Observation state contains NaN or infinity; refusing to produce an action.")
    if not np.isfinite(max_relative_target) or max_relative_target <= 0:
        raise RuntimeError("max_relative_target must be a positive finite scalar.")

    lower = CALIBRATION_BOUNDS_DEG[:, 0]
    upper = CALIBRATION_BOUNDS_DEG[:, 1]
    if ((state < lower) | (state > upper)).any():
        raise RuntimeError(
            "Observation state is outside the frozen Follower calibration range; "
            "refusing Real-to-Sim policy inference."
        )

    calibration_clipped, calibration_mask = clamp_action_fail_closed(raw_action)
    relative_lower = np.maximum(lower, state - max_relative_target)
    relative_upper = np.minimum(upper, state + max_relative_target)
    sent = np.clip(calibration_clipped.astype(np.float64), relative_lower, relative_upper)
    relative_mask = sent != calibration_clipped
    if not np.isfinite(sent).all():
        raise RuntimeError("Relative clipping produced a non-finite action.")

    return {
        "raw_action": np.asarray(raw_action, dtype=np.float32),
        "calibration_clipped_action": calibration_clipped,
        "sent_action": sent.astype(np.float32),
        "calibration_clip_mask": calibration_mask,
        "relative_clip_mask": relative_mask,
    }
