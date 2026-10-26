#!/usr/bin/env python
"""Swap-shift mechanism audit (held-out dataset frames, no training): for each seed / frame bucket, relate ||pred(swap)-pred(true)|| to (a) whether the fine matcher's selected crop changed, (b) the max candidate probability (confidence) under each text.
Discrete card switch = shift is large iff the selected crop changed; blended/unstable = large shifts without a crop change, or low-confidence p."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import g4c_matcher_train as gc  # noqa: E402
from g2_train import FRAMES  # noqa: E402

OD = Path("outputs/eval/safediff_vla_g4d_offline")


@torch.no_grad()
def run(model, d, name="dheld"):
    n = d.d[name]["n"]; P, S, PR = [], [], []
    for t in range(3):
        pr, ss, pp = [], [], []
        for i in range(0, n, 64): a, _, p, s, _ = d.fwd_data(model, name, np.arange(i, min(n, i + 64)), t); pr.append(gc.to_m(a, d).cpu()); ss.append(s.cpu()); pp.append(p.cpu())
        P.append(torch.cat(pr).numpy()); S.append(torch.cat(ss).numpy()); PR.append(torch.cat(pp).numpy())
    return P, S, PR


def main():
    d = gc.Data(); fr = np.array([m["frame"] for m in d.d["dheld"]["meta"]]); cv = d.d["dheld"]["cvalid"].cpu().numpy(); crop = d.d["dheld"]["crop"].cpu().numpy(); res = {}
    for s in (0, 1, 2):
        m = gc.G4cModel("add", 0.2).to("cuda"); m.load_state_dict(torch.load(OD / f"g4d_matcher_s{s}.pt")); m.eval(); P, S, PR = run(m, d); res[s] = {}
        top = [np.where(cv[:, t], S[t], -1e9).argmax(1) for t in range(3)]; sel = [crop[np.arange(len(fr)), t, top[t]] for t in range(3)]
        for k in FRAMES:
            msk = fr == k; sh = np.concatenate([np.linalg.norm((P[t] - P[0])[:, :2], axis=1)[msk] * 100 for t in (1, 2)]); ch = np.concatenate([(sel[t] != sel[0])[msk] for t in (1, 2)]); conf = np.concatenate([PR[t].max(1)[msk] for t in (1, 2)])
            q = np.percentile(sh, [25, 50, 75, 90])
            res[s][k] = {"n": int(len(sh)), "shift_quantiles_25_50_75_90": q.tolist(), "frac_selected_crop_changed": float(ch.mean()), "shift_mean_if_crop_changed": float(sh[ch].mean()) if ch.any() else None, "shift_mean_if_crop_same": float(sh[~ch].mean()) if (~ch).any() else None,
                         "frac_shift_gt5_with_crop_same": float(((sh > 5) & ~ch).mean()), "frac_shift_gt5_with_crop_changed": float(((sh > 5) & ch).mean()), "mean_max_candidate_p_swapped": float(conf.mean())}
            x = res[s][k]; print(f"s{s} {k:4s} shift q25/50/75/90 {np.round(q, 1)} | crop changed {x['frac_selected_crop_changed']:.2f} | mean shift if changed {x['shift_mean_if_crop_changed']} / same {x['shift_mean_if_crop_same']} | >5cm&crop-same {x['frac_shift_gt5_with_crop_same']:.2f} >5cm&changed {x['frac_shift_gt5_with_crop_changed']:.2f} | max-p {x['mean_max_candidate_p_swapped']:.2f}")
    (OD / "g4d_swap_mechanism.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
