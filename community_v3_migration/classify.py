"""Classify each sub-dataset: is it SO-100/101, and what joint encoding is it in?

Detection = robot_type string (recording-time signal) cross-checked against the
per-episode stats min/max (magnitude + exact-boundary saturation). Mismatches are
flagged as `ambiguous` for manual review rather than silently converted.
"""
import json
from pathlib import Path
import numpy as np

# so100/so101 (+ _follower/_bimanual), so_follower, and bimanual bi_so* (bi_so_follower,
# bi_so100_follower, ...; 12-dim).
SO_PREFIXES = ("so100", "so101", "so_", "bi_so")
SO_EXACT: set[str] = set()
# Robots that superficially look SO-like but are NOT in scope for the joint fix:
NEVER_FIX = {"koch", "koch_follower", "koch_bimanual", "moss", "moss_follower"}

RAD_MAX = 3.5     # |val| below this => radians
DEG_MIN = 105.0   # |val| above this => old-convention degrees
SAT_ATOL = 0.5    # closeness to +/-100 / 0 / 100 counted as normalization saturation


def is_so_robot_type(rt: str) -> bool:
    """True if the recorded ``robot_type`` denotes an in-scope SO-100/101 arm."""
    return bool(rt) and (rt.startswith(SO_PREFIXES) or rt in SO_EXACT) and rt not in NEVER_FIX


def load_info(root: Path) -> dict:
    return json.loads((Path(root) / "meta" / "info.json").read_text())


def _global_bounds(root: Path):
    """Per-joint global min/max over action (fallback observation.state), across episodes."""
    lo = hi = None
    key_used = None
    with open(Path(root) / "meta" / "episodes_stats.jsonl") as f:
        for line in f:
            s = json.loads(line)["stats"]
            key = "action" if "action" in s else ("observation.state" if "observation.state" in s else None)
            if key is None:
                continue
            key_used = key
            mn = np.asarray(s[key]["min"], dtype=float)
            mx = np.asarray(s[key]["max"], dtype=float)
            lo = mn if lo is None else np.minimum(lo, mn)
            hi = mx if hi is None else np.maximum(hi, mx)
    return lo, hi, key_used


def encoding_from_bounds(lo, hi, rt: str) -> dict:
    """Detect the SO-arm joint encoding from per-joint global min/max and the robot_type name.

    Layout-agnostic (v2.1 episodes_stats or v3.0 stats.json both reduce to lo/hi here), so it is
    the single source of truth for the degrees_old / degrees_new / normalized / radians decision.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    maxabs = float(np.nanmax(np.abs(np.concatenate([lo, hi]))))
    # saturation on any arm joint (index != gripper) at +/-100, or gripper at 0/100
    n = 6
    sat = False
    for a in range(len(hi) // n):
        arm_hi, arm_lo = hi[a * n:a * n + n], lo[a * n:a * n + n]
        joints_hi, joints_lo = arm_hi[:5], arm_lo[:5]
        grip_hi, grip_lo = arm_hi[5], arm_lo[5]
        sat |= bool(np.any(np.isclose(joints_hi, 100, atol=SAT_ATOL)) or
                    np.any(np.isclose(joints_lo, -100, atol=SAT_ATOL)) or
                    np.isclose(grip_hi, 100, atol=SAT_ATOL) or np.isclose(grip_lo, 0, atol=SAT_ATOL))

    if maxabs <= RAD_MAX:
        enc = "radians"
    elif maxabs > DEG_MIN:
        enc = "degrees_old"
    elif sat:
        enc = "normalized"
    else:
        enc = "degrees_new"

    name_says_new = rt.endswith(("_follower", "_bimanual"))
    ambiguous = (enc == "degrees_old" and name_says_new) or (enc in ("normalized", "degrees_new") and not name_says_new)
    return {"encoding": enc, "maxabs": round(maxabs, 2), "saturates": sat, "ambiguous": ambiguous}


def classify(root) -> dict:
    root = Path(root)
    info = load_info(root)
    rt = info.get("robot_type", "") or ""
    dim = (info.get("features", {}).get("action", {}).get("shape") or [None])[0]
    out = {"root": str(root), "robot_type": rt, "action_dim": dim,
           "codebase_version": info.get("codebase_version"), "ambiguous": False}

    is_so = is_so_robot_type(rt)
    if not is_so:
        return {**out, "is_so": False, "encoding": "non_so"}

    lo, hi, key_used = _global_bounds(root)
    if lo is None:
        return {**out, "is_so": True, "encoding": "unknown", "ambiguous": True,
                "note": "no action/state stats found"}

    return {**out, "is_so": True, "stats_key": key_used, **encoding_from_bounds(lo, hi, rt)}
