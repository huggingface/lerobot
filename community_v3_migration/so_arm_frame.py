"""SO-100/101 joint-frame conversion to physical degrees (post-#777 convention).

Two calibration-free branches + one that needs an assumed canonical range:

  * degrees_old  (bare robot_type `so100`/`so101`, |vals|>~180): PR #3879 old->new
                 convention (sign flip shoulder_lift, +90 deg shoulder_lift/elbow_flex).
                 EXACT.
  * degrees_new  (`*_follower` recorded with use_degrees=True, not saturated): already
                 degrees. EXACT.
  * normalized   (`*_follower`, -100..100 joints / 0..100 gripper, saturates at bounds):
                 5 arm joints are mid-range-zero, only the SCALE is missing (per-robot
                 range_min/max not stored) -> use assumed canonical spans below. APPROXIMATE.
                 The gripper (0..100) is kept in its native frame, matching degrees_new.
  * radians      -> untouched.

Joint order per arm: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
Bimanual (12-dim) tiles the 6-joint block twice.
"""
import numpy as np

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# --- PR #3879 (degrees). old(community frame) <-> new(v3.0 / post-#777) frame. ---
SIGNS = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
OFFSETS_DEG = np.array([0.0, 90.0, 90.0, 0.0, 0.0, 0.0], dtype=np.float64)

# --- Canonical per-joint spans (DEGREES) used ONLY to invert the -100..100 normalization of
#     the 5 arm joints (RANGE_M100_100) when per-robot calibration is unavailable: normalized
#     +/-100 -> +/-HALF_RANGE. The gripper (RANGE_0_100) is left in its native 0..100 frame in
#     every SO dataset, so it needs no canonical span. THESE ARE PLACEHOLDERS — run
#     calibrate_canonical_ranges.py and paste the fitted values here before a production run. ---
CANON_HALF_RANGE_DEG = np.array([100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float64)  # 5 arm joints
CANON_IS_CALIBRATED = False  # flipped to True once you paste fitted values


def _convert_arm(x: np.ndarray, encoding: str) -> np.ndarray:
    """x: (..., 6) for a single SO arm -> degrees (..., 6)."""
    x = np.asarray(x, dtype=np.float64)
    if encoding == "radians":
        return x
    if encoding == "degrees_old":
        return SIGNS * (x - OFFSETS_DEG)
    if encoding == "degrees_new":
        return x
    if encoding == "normalized":
        new_deg = np.array(x, dtype=np.float64)
        new_deg[..., :5] = (x[..., :5] / 100.0) * CANON_HALF_RANGE_DEG
        # gripper is RANGE_0_100 in every SO dataset (including use_degrees=True / degrees_new),
        # so it is already frame-consistent and must be left untouched, not remapped to +/-deg.
        return new_deg
    raise ValueError(f"unknown encoding: {encoding!r}")


def to_degrees(arr, encoding: str, n_joints_per_arm: int = 6) -> np.ndarray:
    """arr: (..., D) with D a multiple of 6. Returns float32 degrees, same shape."""
    arr = np.asarray(arr, dtype=np.float64)
    d = arr.shape[-1]
    if d % n_joints_per_arm != 0:
        raise ValueError(f"action/state dim {d} is not a multiple of {n_joints_per_arm}")
    if encoding == "normalized" and not CANON_IS_CALIBRATED:
        raise RuntimeError(
            "CANON ranges are placeholders. Run calibrate_canonical_ranges.py and set "
            "CANON_* + CANON_IS_CALIBRATED=True before converting 'normalized' datasets to degrees."
        )
    out = np.empty_like(arr)
    for a in range(d // n_joints_per_arm):
        sl = slice(a * n_joints_per_arm, (a + 1) * n_joints_per_arm)
        out[..., sl] = _convert_arm(arr[..., sl], encoding)
    return out.astype(np.float32)
