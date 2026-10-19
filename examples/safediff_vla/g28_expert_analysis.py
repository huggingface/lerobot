#!/usr/bin/env python
"""G28 STEP 1 analysis: what visual regime does the scripted expert use for book / drink? (successful expert episodes) -> tables, trajectory plots (target-relative EE path), visibility curves, wrist/global montage of the 40 steps before the close."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/eval/safediff_vla_g28"); TASKS = ["select_book", "select_drink"]; FWD = np.array([0.0, 1.0, 0.0])


def main():
    for t in TASKS:
        E = [e for e in pickle.load(open(OUT / t / "expert" / "expert_audit.pkl", "rb")) if e["success"] and e["t_close"] is not None]; print(f"\n===== {t[7:]}: successful expert episodes {len(E)} =====")
        fig, ax = plt.subplots(2, 3, figsize=(18, 9)); rel_all = []
        rows = []
        for e in E:
            S = e["steps"]; tc = e["t_close"]; kp = S[tc]["kp"]; ee = np.array([s["ee_p"] for s in S]); rel = ee - kp; prep = kp - 0.1 * FWD; dprep = np.linalg.norm(ee - prep, axis=1); tprep = int(np.argmin(dprep[:tc + 1]))
            seg = ee[:tc + 1]; path = np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)); strt = np.linalg.norm(seg[-1] - seg[0]); line = seg[0] + np.outer(np.linspace(0, 1, len(seg)), seg[-1] - seg[0]); dev = np.max(np.linalg.norm(seg - line, axis=1))
            wv = np.array([s["wrist_vis"] for s in S]); fv = np.array([s["front_vis"] for s in S]); wuv = np.array([s["wrist_uv"] if s["wrist_uv"] else (np.nan, np.nan, np.nan) for s in S]); last = slice(max(tc - 20, 0), tc)
            ctr = np.hypot(wuv[last, 0] - 240, wuv[last, 1] - 240); depth = wuv[last, 2]
            # approach-direction at the end: displacement over the last 10 steps before close
            v10 = ee[tc] - ee[max(tc - 10, 0)]; cosf = float(v10 @ FWD / (np.linalg.norm(v10) + 1e-9))
            rows.append({"n": e["n"], "tc": tc, "prep_d_cm": float(dprep[tprep]) * 100, "t_prep": tprep, "straight": strt / path, "dev_cm": dev * 100, "front_vis_close": int(fv[tc]), "front_vis_min20": int(fv[last].min()), "wrist_vis_close": int(wv[tc]), "wrist_vis_min20": int(wv[last].min()), "wrist_vis_frac20": float(np.mean(wv[last] > 20)), "uv_dev_close": float(np.hypot(wuv[tc, 0] - 240, wuv[tc, 1] - 240)) if not np.isnan(wuv[tc, 0]) else np.nan, "uv_dev_mean20": float(np.nanmean(ctr)), "uv_dev_sd20": float(np.nanstd(ctr)), "depth_close_cm": float(wuv[tc, 2]) * 100 if not np.isnan(wuv[tc, 2]) else np.nan, "cos_fwd": cosf, "front_vis_frac_all": float(np.mean(fv[:tc] > 20))})
            ax[0, 0].plot(rel[:tc + 1, 1] * 100, rel[:tc + 1, 2] * 100); ax[0, 1].plot(rel[:tc + 1, 0] * 100, rel[:tc + 1, 1] * 100); tt = np.arange(-tc, 1); ax[0, 2].plot(tt, np.linalg.norm(rel[:tc + 1], axis=1) * 100); ax[1, 0].plot(tt, wv[:tc + 1]); ax[1, 1].plot(tt, fv[:tc + 1]); ax[1, 2].plot(np.arange(-20, 0), ctr)
        for a, ttl, xl, yl in ((ax[0, 0], "EE path relative to target keypoint, side (y fwd vs z), cm", "y (toward shelf/fridge) cm", "z cm"), (ax[0, 1], "EE path relative to target keypoint, top-down (x vs y), cm", "x cm", "y cm"), (ax[0, 2], "EE-target distance vs steps to close (cm)", "step to close", "cm"), (ax[1, 0], "target visible pixels, WRIST cam (240px)", "step to close", "px"), (ax[1, 1], "target visible pixels, FRONT cam (240px)", "step to close", "px"), (ax[1, 2], "target projection distance from wrist-image centre, last 20 steps (480px scale)", "step to close", "px")): a.set_title(ttl, fontsize=9); a.set_xlabel(xl, fontsize=8); a.set_ylabel(yl, fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / f"g28_expert_{t}.png", dpi=50); plt.close(fig)
        keys = ["n", "tc", "t_prep", "prep_d_cm", "straight", "dev_cm", "front_vis_frac_all", "front_vis_min20", "front_vis_close", "wrist_vis_frac20", "wrist_vis_min20", "wrist_vis_close", "uv_dev_mean20", "uv_dev_sd20", "uv_dev_close", "depth_close_cm", "cos_fwd"]
        print("   metric (mean [min-max] over expert episodes):")
        for k in keys: v = np.array([r[k] for r in rows], float); print(f"     {k:22s} {np.nanmean(v):9.2f} [{np.nanmin(v):.2f} - {np.nanmax(v):.2f}]")
        # montage
        e = E[0]; ts = sorted(e["frames"]); ts = ts[-8:]; fig, ax = plt.subplots(2, len(ts), figsize=(3 * len(ts), 6), squeeze=False)
        for c, tt_ in enumerate(ts): w, f = e["frames"][tt_]; ax[0, c].imshow(w); ax[0, c].set_title(f"wrist t{tt_ - e['t_close']:+d}", fontsize=8); ax[1, c].imshow(f); ax[1, c].set_title("front", fontsize=8); ax[0, c].axis("off"); ax[1, c].axis("off")
        fig.tight_layout(); fig.savefig(OUT / f"g28_expert_montage_{t}.png", dpi=45); plt.close(fig)


if __name__ == "__main__":
    main()
