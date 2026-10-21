#!/usr/bin/env python
"""StagePlan-v2 TRAIN labels: ONLY the variables the historical oracle interventions supplied (grasp pose, approach, pre-grasp
distance, global waypoint). NO P2/P3 / post-grasp supervision (the learned v2 decoder keeps post-grasp control).

Built on `stageplan_labels.extract` events (close transition, motion dwell, polyline corners); task-agnostic.
  grasp_xyz/euler : P1 = expert commanded EE pose on arrival, immediately before CLOSE (robot-base frame, scipy 'xyz' euler)
  approach_dir    : unit(P1 - P0)   (P0 = last path corner >= 5 cm before P1)
  pre_dist        : |P1 - P0|
  wp_rel          : W - P1, W = expert commanded EE at the FIRST step within WP_RADIUS (25 cm, inherited from G28's w25) of P1
                    (if the demo starts inside 25 cm, W = the first pose). waypoint_valid is NOT defined: the audit found waypoint
                    'distinctness' (deviation from the straight start->P0 chord) does not separate drink from other families.
Usage: uv run python examples/safediff_vla/stageplan_v2_labels.py   # -> outputs/stageplan/labels_v2.pkl
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import stageplan_labels as SL  # noqa: E402

WP_RADIUS = 0.25


def labels_v2(a: np.ndarray) -> dict:
    lab = SL.extract(a)
    if not lab["valid"]: return {"valid": False, "reason": lab["reason"]}
    P0, P1 = lab["P0"], lab["P1"]; v = P1[:3] - P0[:3]; r = float(np.linalg.norm(v))
    ee = a[: lab["t1"] + 1, :3]; d = np.linalg.norm(ee - P1[:3], axis=1); i = int(np.argmax(d <= WP_RADIUS)) if d[0] > WP_RADIUS else 0
    return {"valid": True, "grasp_xyz": P1[:3].copy(), "grasp_euler": P1[3:].copy(), "approach_dir": v / max(r, 1e-9), "pre_dist": r, "wp_rel": ee[i] - P1[:3], "t1": lab["t1"], "tc": lab["tc"]}


def main() -> None:
    df = pd.read_pickle(SL.OUT / "df6.pkl"); out = {}
    for ep, e in df.groupby("episode_index"):
        l = labels_v2(np.stack(e.action.values).astype(np.float64)); l["fam"] = e.fam.iloc[0]; out[int(ep)] = l
    pickle.dump(out, open(SL.OUT / "labels_v2.pkl", "wb")); V = [l for l in out.values() if l["valid"]]
    print("episodes", len(out), "valid", len(V))
    for f in sorted({l["fam"] for l in V}):
        X = [l for l in V if l["fam"] == f]
        print(f"  {f:15s} pre_dist {np.median([l['pre_dist'] for l in X]) * 100:.1f} cm | approach_dir mean {np.round(np.mean([l['approach_dir'] for l in X], 0), 2).tolist()} | wp_rel median(cm) {np.round(np.median([l['wp_rel'] for l in X], 0) * 100, 1).tolist()}")


if __name__ == "__main__":
    main()
