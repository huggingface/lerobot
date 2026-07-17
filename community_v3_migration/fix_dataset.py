"""Rewrite observation.state / action to degrees in a LOCAL v2.1 SO-arm dataset, then
regenerate meta/episodes_stats.jsonl (action & state only; other features preserved).
Run this BEFORE the stock v2.1->v3.0 converter so its stats aggregation stays correct.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import so_arm_frame
from classify import classify, load_info, so_joint_count

VALUE_COLS = ("observation.state", "action")


def _stack(col_values) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in col_values])  # (N, D)


def _set_robot_type(root: Path, robot_type: str) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["robot_type"] = robot_type
    info_path.write_text(json.dumps(info, indent=4))


def _rewrite_parquet(root: Path, encoding: str, so_dims: dict) -> None:
    for pq in sorted((root / "data").glob("*/*.parquet")):
        df = pd.read_parquet(pq)
        changed = False
        for col in VALUE_COLS:
            n = so_dims.get(col, 0)
            if col in df.columns and n:
                full = _stack(df[col].values)  # (N, D)
                full[:, :n] = so_arm_frame.to_degrees(full[:, :n], encoding, n_joints_per_arm=6)
                df[col] = list(full.astype(np.float32))
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


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def reconcile_episode_count(root) -> str | None:
    """When the data files and video files agree on an episode count N but the metadata lists a
    different count, rewrite the metadata (episodes.jsonl, episodes_stats.jsonl, info.json) to N.

    Only the safe direction is handled: trimming metadata that lists MORE episodes than actually
    exist. If the data itself is non-contiguous, the videos disagree with the data, or the metadata
    lists FEWER episodes than the data (which would require fabricating per-episode stats), nothing
    is changed and the stock converter's mismatch error is left to surface. Returns a note on fix."""
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())

    data_idx = sorted(int(p.stem.split("_")[-1]) for p in (root / "data").glob("*/episode_*.parquet"))
    n = len(data_idx)
    if n == 0 or data_idx != list(range(n)):
        return None

    vkeys = [k for k, f in info.get("features", {}).items() if f.get("dtype") == "video"]
    for k in vkeys:
        if len(list((root / "videos").glob(f"*/{k}/episode_*.mp4"))) != n:
            return None  # data and videos disagree -> out of scope for this fix

    eps_path = root / "meta" / "episodes.jsonl"
    stats_path = root / "meta" / "episodes_stats.jsonl"
    eps, stats = _read_jsonl(eps_path), _read_jsonl(stats_path)
    if len(eps) == n and len(stats) == n:
        return None

    eps_keep = [e for e in eps if e.get("episode_index", -1) < n]
    stats_keep = [s for s in stats if s.get("episode_index", -1) < n]
    if len(eps_keep) != n or len(stats_keep) != n:
        return None  # metadata is missing episodes present in the data -> can't safely fabricate

    dropped = max(len(eps), len(stats)) - n
    _write_jsonl(eps_path, eps_keep)
    _write_jsonl(stats_path, stats_keep)
    info["total_episodes"] = n
    info["total_frames"] = int(sum(e.get("length", 0) for e in eps_keep))
    if "total_videos" in info:
        info["total_videos"] = n * len(vkeys)
    info["splits"] = {"train": f"0:{n}"}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    return f"metadata episode count reconciled to {n} (data & videos agree; dropped {dropped} stale meta entries)"


def fix_dataset_in_place(root) -> dict:
    """Returns the classification dict augmented with the action taken."""
    root = Path(root)
    cls = classify(root)
    if cls.get("mislabeled_so"):
        # robot_type claims SO but the joints prove otherwise (wrong dim or non-SO names).
        # Relabel to 'unknown' and migrate structurally rather than degrees-converting on a
        # false assumption; the joint values are left exactly as recorded.
        _set_robot_type(root, "unknown")
        return {**cls, "robot_type": "unknown", "converted": False,
                "action": f"structural v2.1->v3.0 only; robot_type relabeled '{cls.get('robot_type')}'"
                          "->'unknown' (joints don't match a 6-DOF SO arm), joint values left unchanged"}
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
    # drop stray files that would otherwise be uploaded
    for junk in (root / "meta").glob("info.json.bak"):
        junk.unlink()
    info = load_info(root)
    so_dims = {c: so_joint_count(info, c) for c in VALUE_COLS}
    _rewrite_parquet(root, enc, so_dims)
    _regen_episode_stats(root)
    full_dims = {c: (info.get("features", {}).get(c, {}).get("shape") or [0])[0] for c in VALUE_COLS}
    partial = any(0 < so_dims[c] < full_dims[c] for c in VALUE_COLS)
    tail = " (leading SO joints only; trailing non-joint columns left unchanged)" if partial else ""
    return {**cls, "converted": True,
            "action": f"structural v2.1->v3.0 + joint values converted ({enc} -> degrees){tail}"}
