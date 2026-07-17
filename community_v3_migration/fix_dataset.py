"""Rewrite observation.state / action to degrees in a LOCAL v2.1 SO-arm dataset, then
regenerate meta/episodes_stats.jsonl (action & state only; other features preserved).
Run this BEFORE the stock v2.1->v3.0 converter so its stats aggregation stays correct.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import so_arm_frame
from classify import classify, load_info

VALUE_COLS = ("observation.state", "action")


def _stack(col_values) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in col_values])  # (N, D)


def _set_robot_type(root: Path, robot_type: str) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["robot_type"] = robot_type
    info_path.write_text(json.dumps(info, indent=4))


def _rewrite_parquet(root: Path, encoding: str) -> None:
    for pq in sorted((root / "data").glob("*/*.parquet")):
        df = pd.read_parquet(pq)
        changed = False
        for col in VALUE_COLS:
            if col in df.columns:
                conv = so_arm_frame.to_degrees(_stack(df[col].values), encoding, n_joints_per_arm=6)
                df[col] = list(conv.astype(np.float32))
                changed = True
        if changed:
            df.to_parquet(pq, index=False)


def _regen_episode_stats(root: Path) -> None:
    stats_path = root / "meta" / "episodes_stats.jsonl"
    orig = {}
    with open(stats_path) as f:
        for line in f:
            e = json.loads(line)
            orig[e["episode_index"]] = e
    for pq in sorted((root / "data").glob("*/*.parquet")):
        df = pd.read_parquet(pq)
        for ep in np.unique(df["episode_index"].values):
            ep = int(ep)
            sub = df[df["episode_index"] == ep]
            entry = orig.get(ep)
            if entry is None:
                continue
            for col in VALUE_COLS:
                if col in sub.columns:
                    a = _stack(sub[col].values)  # (n, D)
                    entry["stats"][col] = {
                        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
                        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
                        "count": [int(a.shape[0])],
                    }
    with open(stats_path, "w") as f:
        for ep in sorted(orig):
            f.write(json.dumps(orig[ep]) + "\n")


def fix_dataset_in_place(root) -> dict:
    """Returns the classification dict augmented with the action taken."""
    root = Path(root)
    cls = classify(root)
    enc = cls.get("encoding")
    if not cls.get("is_so") or enc in ("radians", "unknown", "non_so"):
        reason = {
            "non_so": "not an SO-100/101 dataset",
            "radians": "SO-arm joints already in radians",
            "unknown": "SO-arm but joint encoding could not be determined",
        }.get(enc, "no joint conversion applicable")
        return {**cls, "converted": False,
                "action": f"structural v2.1->v3.0 only ({reason}); joint values left unchanged"}
    if enc == "normalized" and not so_arm_frame.CANON_IS_CALIBRATED:
        # Without per-robot calibration the un-normalization is an identity (placeholder
        # spans == 100), so rewriting is pointless. Keep the normalized values as-is and let
        # the dataset card flag them APPROXIMATE instead.
        return {**cls, "converted": False,
                "action": "structural v2.1->v3.0 only; joint values kept in normalized units "
                          "(-100..100 / 0..100), NOT converted to degrees (uncalibrated -> APPROXIMATE)"}
    feats = load_info(root).get("features", {})
    dims = [feats[c]["shape"][0] for c in VALUE_COLS if c in feats and feats[c].get("shape")]
    if any(d % 6 != 0 for d in dims):
        # Not a plain stack of 6-joint SO arms (e.g. a 7-joint variant): the degrees mapping
        # doesn't apply. Keep the original robot_type but flag it '_nonstandard' so the SO
        # lineage is preserved while making clear it isn't a canonical 6-DOF arm.
        rt = cls.get("robot_type") or "so"
        new_rt = rt if rt.endswith("_nonstandard") else f"{rt}_nonstandard"
        _set_robot_type(root, new_rt)
        return {**cls, "robot_type": new_rt, "converted": False,
                "action": f"structural v2.1->v3.0 only; joint dims {dims} not a multiple of 6 "
                          f"(non-standard arm), robot_type set to '{new_rt}', joint values left unchanged"}
    # drop stray files that would otherwise be uploaded
    for junk in (root / "meta").glob("info.json.bak"):
        junk.unlink()
    _rewrite_parquet(root, enc)
    _regen_episode_stats(root)
    return {**cls, "converted": True,
            "action": f"structural v2.1->v3.0 + joint values converted ({enc} -> degrees)"}
