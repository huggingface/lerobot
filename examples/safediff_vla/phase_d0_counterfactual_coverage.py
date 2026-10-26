#!/usr/bin/env python
"""PHASE D0 step F (part 2): counterfactual coverage in the TRAINING data. Does the dataset contain
(same / approximately same initial scene layout) + (different referent) pairs -- the supervision a model
needs to learn "language picks WHICH object" -- or is every episode a fresh random layout?

Scene similarity (defined a priori, not tuned): frame 0 of the front and second cameras, 96x96 RGB; the arm is at
its fixed home pose at frame 0, so any difference is the object layout (plus codec noise). Metric
M(i,j) = max over a 16x16 grid of 6x6-px blocks and channels of |block-mean(i) - block-mean(j)|, then max over
the two cameras ("largest local appearance change"). A moved object changes some block by tens of gray levels;
codec noise is a few. Reported at three FIXED cut-offs (6 / 12 / 24 gray levels) -- 12 is the primary one; they are
sensitivity levels, not a tuned optimum. A counterfactual pair = same task, M <= cut-off, DIFFERENT referent.
Also reported: exact pixel duplicates, and the distribution of each episode's nearest-neighbour M (a gap near 0
would show repeated layouts; none -> every layout is unique).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/eval/safediff_vla_phase_d0_training_data_audit")
CUTS = (6, 12, 24)


def block_means(x):  # x [n,cam,96,96,3] -> [n,cam,16,16,3] float
    n, c = x.shape[:2]
    return x.reshape(n, c, 16, 6, 16, 6, 3).astype(np.float32).mean(axis=(3, 5))


def main():
    z = np.load(OUT / "first_frames_96.npz", allow_pickle=True)
    imgs, ep_ids, tasks = z["imgs"], z["episode"], z["task"]
    ep = pd.read_json(OUT / "episodes.json").set_index("episode")
    refs = ep.loc[ep_ids, "referent"].to_numpy(); instr = ep.loc[ep_ids, "instruction"].to_numpy()
    res = {}
    for task in sorted(set(tasks)):
        idx = np.where(tasks == task)[0]; n = len(idx)
        B = block_means(imgs[idx]).reshape(n, 2, -1)  # [n,cam,768]
        M = np.zeros((n, n), np.float32)
        for i in range(n):
            M[i] = np.abs(B - B[i]).max(axis=2).max(axis=1)  # max over blocks*channels, then over cameras
        np.fill_diagonal(M, np.inf)
        r_ = refs[idx]; diff_ref = r_[:, None] != r_[None, :]
        exact = int(sum((imgs[idx[i]] == imgs[idx[j]]).all() for i in range(n) for j in np.where(M[i] <= 1.0)[0] if j > i))
        nn = M.min(axis=1)
        out = {"n_episodes": n, "exact_pixel_duplicate_pairs": exact,
               "nearest_neighbour_M_pctiles_0_5_25_50_75_95": np.percentile(nn, [0, 5, 25, 50, 75, 95]).round(1).tolist()}
        for c in CUTS:
            same_layout = M <= c
            cf = same_layout & diff_ref
            out[f"cut_{c}"] = {"episodes_with_a_same_layout_partner": int(same_layout.any(1).sum()),
                               "same_layout_pairs": int(same_layout.sum() // 2),
                               "counterfactual_pairs(same layout, different referent)": int(cf.sum() // 2),
                               "episodes_with_counterfactual_partner": int(cf.any(1).sum()),
                               "frac_episodes_with_counterfactual_partner": float(cf.any(1).mean())}
        res[task] = out
        print(task, out["nearest_neighbour_M_pctiles_0_5_25_50_75_95"], {c: out[f"cut_{c}"]["episodes_with_counterfactual_partner"] for c in CUTS}, "exact dup pairs", exact, flush=True)
    (OUT / "counterfactual_coverage.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
