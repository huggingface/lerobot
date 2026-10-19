"""Shared G1 evaluation metrics on counterfactual scenes (XY primary). Same definitions as G0.6/G0.6b:
R_follow = sum(dPred . dGT) / sum(|dGT|^2) over all within-scene referent pairs (ratio of sums), per-pair medians of
R_follow / R_mag / R_dir, plus predicted shift, localisation error to the referent's own anchor, referent-selection
accuracy (nearest candidate anchor to the prediction == the instructed one) and margin. Scene-bootstrap CI."""
from __future__ import annotations

from itertools import combinations

import numpy as np


def metrics(scenes: list[dict], n_boot: int = 2000, seed: int = 0, pair_ok=None, query_ok=None) -> dict:
    """scenes: [{"anchors": {name: xyz}, "preds": {name: xyz}}] (robot frame, metres)."""
    num = den = 0.0; foll, mag, dirs, shifts, err, acc, marg, nums, dens = [], [], [], [], [], [], [], [], []
    for s in scenes:
        n_s = d_s = 0.0
        for a, b in combinations(list(s["anchors"]), 2):
            if pair_ok is not None and not pair_ok(a, b):
                continue
            g = (np.asarray(s["anchors"][b]) - np.asarray(s["anchors"][a]))[:2]; p = (np.asarray(s["preds"][b]) - np.asarray(s["preds"][a]))[:2]
            n_s += float(p @ g); d_s += float(g @ g); foll.append(float(p @ g / (g @ g))); mag.append(float(np.linalg.norm(p) / np.linalg.norm(g)))
            dirs.append(float(p @ g / (np.linalg.norm(p) * np.linalg.norm(g) + 1e-12))); shifts.append(float(np.linalg.norm(p)))
        nums.append(n_s); dens.append(d_s)
        for c in s["anchors"]:
            if query_ok is not None and not query_ok(c):
                continue
            d = {k: np.linalg.norm((np.asarray(s["preds"][c]) - np.asarray(v))[:2]) for k, v in s["anchors"].items()}
            err.append(d[c]); acc.append(min(d, key=d.get) == c); others = [v for k, v in d.items() if k != c]
            marg.append(min(others) - d[c])
    nums, dens = np.array(nums), np.array(dens); rng = np.random.default_rng(seed); n = len(nums)
    boot = [nums[i].sum() / dens[i].sum() for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return {"n_scenes": n, "n_samples": len(err), "n_pairs": len(foll),
            "R_follow_xy_pooled": float(nums.sum() / dens.sum()), "R_follow_ci95_scene_bootstrap": np.percentile(boot, [2.5, 97.5]).round(3).tolist(),
            "R_follow_pair_median": float(np.median(foll)), "R_mag_pair_median": float(np.median(mag)), "R_dir_pair_median": float(np.median(dirs)),
            "pred_shift_xy_cm_mean": float(np.mean(shifts) * 100),
            "gt_shift_xy_cm_mean": float(np.mean([np.linalg.norm((np.asarray(s['anchors'][b]) - np.asarray(s['anchors'][a]))[:2]) for s in scenes for a, b in combinations(list(s['anchors']), 2) if pair_ok is None or pair_ok(a, b)]) * 100),
            "loc_err_xy_cm_mean": float(np.mean(err) * 100), "loc_err_xy_cm_median": float(np.median(err) * 100),
            "selection_acc": float(np.mean(acc)), "selection_chance": float(np.mean([1 / len(s["anchors"]) for s in scenes for c in s["anchors"] if query_ok is None or query_ok(c)])),
            "selection_margin_cm_median": float(np.median(marg) * 100)}
