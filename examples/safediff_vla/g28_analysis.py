#!/usr/bin/env python
"""G28 analysis: arms A GLOBAL-POINT | B GLOBAL-POSE | C GLOBAL-WAYPOINT+POSE | D WAYPOINT+WRIST-LOCAL (and D2 if run), book and drink, seeds 3000-3004. Stage metrics (collision, visibility, reach, alignment at close, grasp acquisition, retention, success), last-20-step target-relative error curves, representative trajectory plots and wrist/front montages."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from g25_analysis import events  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g28"); TASKS = ["select_book", "select_drink"]; ARMS = ["A", "B", "C", "D", "D2"]; NAME = {"A": "GLOBAL-POINT", "B": "GLOBAL-POSE", "C": "WAYPOINT+POSE", "D": "WAYPOINT+WRIST", "D2": "FROZEN-WRIST"}


def load(t, a):
    p = OUT / t / a / "g28_trace.pkl"
    if not p.exists(): return None
    L = pickle.load(open(p, "rb")); eps = []
    for i, e in enumerate(L["episodes"]):
        S = [s for s in L["steps"] if s["ep"] == i]; rec = e["rec"]; ci = next((k for k, r in enumerate(rec) if r["reactive_close_event"]), None); eps.append((S, e, ci))
    return L, eps


def ep_metrics(S, e, ci):
    rec = e["rec"]; tgt = e["target"]; n = len(S); tc = max(ci - 1, 0) if ci is not None else None; end = tc if tc is not None else n - 1; d = np.array([np.linalg.norm(s["ee"] - s["kp"]) for s in S]); m = {"seed": e["seed"], "success": bool(e["success"]), "close": tc is not None}
    t0 = np.array(rec[0]["target_pos"]); tp = np.array([r["target_pos"] for r in rec]); c0 = rec[0]["candidates_world"]; oth = [k for k in c0 if k != tgt]
    disp_t = np.linalg.norm(tp - t0, axis=1); disp_o = np.array([[np.linalg.norm(np.array(r["candidates_world"][k]) - np.array(c0[k])) for k in oth] for r in rec]) if oth else np.zeros((len(rec), 0)); pe = end + 1
    m["pre_t_cm"] = disp_t[:pe].max() * 100; m["pre_o_cm"] = (disp_o[:pe].max() * 100 if disp_o.size else 0.0); m["collision"] = bool(m["pre_t_cm"] > 1.5 or m["pre_o_cm"] > 1.5); m["oth_max_cm"] = disp_o.max() * 100 if disp_o.size else 0.0
    ph0 = [s for s in S[:end + 1] if s["phase"] == 0]; last = S[max(end - 20, 0):end + 1]; m["wrist_vis20"] = float(np.mean([s["wrist_vis"] > 20 for s in last])); m["front_vis20"] = float(np.mean([s["front_vis"] > 20 for s in last]))
    m["min_d_cm"] = float(d[:end + 1].min() * 100); m["reach5"] = m["min_d_cm"] < 5; m["reach3"] = m["min_d_cm"] < 3
    if tc is not None:
        s0 = S[tc]; rel = (s0["ee"] - s0["kp"]) * 100; m["lat_close_cm"] = float(np.hypot(rel[0], rel[2])); m["axial_behind_cm"] = float(-rel[1]); m["d_close_cm"] = float(np.linalg.norm(rel))
        ev = events(S, tc); w3 = S[min(tc + 3, n - 1)]["width"] * 1000; m["w3"] = w3; m["between"] = w3 > 10; m["ncon"] = float(np.mean([S[min(tc + k, n - 1)]["ncon"] for k in range(1, 11)])); m["bilat"] = m["ncon"] >= 2; g = np.array([s["grasped"] for s in S]); tg = ev["t_grasp"]
        m["held10"] = bool(tg is not None and tg + 10 < n and g[tg:tg + 10].all()); m["held30"] = bool(tg is not None and tg + 30 < n and g[tg:tg + 30].all()); m["post_move_cm"] = float(disp_t[tc:].max() * 100); m["lift"] = ev["t_lift"] is not None; m["held_steps"] = ev["held_steps"]
    ee = np.array([s["ee"] for s in S[:end + 1]]); m["path_cm"] = float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1)) * 100)
    rel = np.array([(s["ee"] - s["kp"]) * 100 for s in S[max(end - 20, 0):end + 1]]); m["curve_lat"] = np.hypot(rel[:, 0], rel[:, 2]); m["curve_ax"] = -rel[:, 1]; return m


def main():
    R = {}
    for t in TASKS:
        for a in ARMS:
            x = load(t, a)
            if x: R[(t, a)] = [ep_metrics(*it) for it in x[1]]
    for t in TASKS:
        print(f"\n================ {t[7:].upper()} ================")
        print(f"{'arm':16s}{'n':>3s}{'collision pre-close':>20s}{'target moved':>13s}{'other moved':>12s}{'wrist vis last20':>17s}{'front vis':>10s}{'reach<5':>8s}{'reach<3':>8s}{'close':>6s}{'lat@close':>10s}{'axial gap':>10s}{'between@+3':>11s}{'bilat':>6s}{'held10':>7s}{'held30':>7s}{'lift':>5s}{'native':>7s}{'path cm':>8s}")
        for a in ARMS:
            if (t, a) not in R: continue
            r = R[(t, a)]; n = len(r); f = lambda k: sum(bool(x.get(k)) for x in r); cl = [x for x in r if x["close"]]; mm = lambda k, src=r: np.mean([x[k] for x in src]) if src else float("nan")
            print(f"{a + ' ' + NAME[a]:16s}{n:3d}{f('collision'):>17d}/{n}{sum(x['pre_t_cm'] > 1.5 for x in r):>10d}/{n}{sum(x['pre_o_cm'] > 1.5 for x in r):>9d}/{n}{mm('wrist_vis20'):17.2f}{mm('front_vis20'):10.2f}{f('reach5'):>5d}/{n}{f('reach3'):>5d}/{n}{len(cl):>3d}/{n}{mm('lat_close_cm', cl):10.1f}{mm('axial_behind_cm', cl):10.1f}{sum(x['between'] for x in cl):>8d}/{len(cl)}{sum(x['bilat'] for x in cl):>3d}/{len(cl)}{sum(x['held10'] for x in cl):>4d}/{len(cl)}{sum(x['held30'] for x in cl):>4d}/{len(cl)}{sum(x['lift'] for x in cl):>2d}/{len(cl)}{f('success'):>4d}/{n}{mm('path_cm'):8.0f}")
        print("   per episode (seed: min dist cm | lateral/axial gap at close cm | between fingers@+3 (w mm) | held30 | pre-close target/other displacement cm | native):")
        for a in ARMS:
            if (t, a) in R: print(f"   {a:2s} " + " ; ".join(f"{x['seed']}: {x['min_d_cm']:.1f}|" + (f"{x['lat_close_cm']:.1f}/{x['axial_behind_cm']:.1f}|{int(x['between'])}(w{x['w3']:.0f})|{int(x['held30'])}" if x['close'] else "noclose|-|-") + f"|{x['pre_t_cm']:.1f}/{x['pre_o_cm']:.1f}|{int(x['success'])}" for x in R[(t, a)]))
        print("   last-20-step target-relative error before the close (mean over episodes; cm at steps -20,-15,-10,-5,0): lateral | axial gap")
        for a in ARMS:
            if (t, a) not in R: continue
            cs = [x for x in R[(t, a)] if len(x["curve_lat"]) == 21]
            if cs: print(f"   {a:2s} lat " + " ".join(f"{np.mean([x['curve_lat'][i] for x in cs]):5.1f}" for i in (0, 5, 10, 15, 20)) + " | axial " + " ".join(f"{np.mean([x['curve_ax'][i] for x in cs]):5.1f}" for i in (0, 5, 10, 15, 20)) + f"  (n={len(cs)})")
    # trajectory plots
    fig, ax = plt.subplots(2, 4, figsize=(24, 10))
    for r_, t in enumerate(TASKS):
        for c, a in enumerate(["A", "B", "C", "D"]):
            x = load(t, a)
            if not x: continue
            for S, e, ci in x[1]:
                tc = max(ci - 1, 0) if ci is not None else len(S) - 1; rel = np.array([(s["ee"] - s["kp"]) * 100 for s in S[:tc + 1]]); ax[r_, c].plot(rel[:, 1], rel[:, 2], label=f"seed {e['seed']}"); ax[r_, c].plot(rel[-1, 1], rel[-1, 2], "ko", ms=4)
            ax[r_, c].axhline(0, color="gray", lw=0.5); ax[r_, c].axvline(0, color="gray", lw=0.5); ax[r_, c].set_title(f"{t[7:]} {a} {NAME[a]}: EE path rel. to keypoint (y toward target, z), cm; dot = close", fontsize=8); ax[r_, c].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "g28_trajectories.png", dpi=40); plt.close(fig)
    # montage: for each task, B / C / D first seed with a close: wrist+front frames from close-25 .. close+10
    for t in TASKS:
        rows = []
        for a in ["B", "C", "D"]:
            p = OUT / t / a / "g28_frames.pkl"; x = load(t, a)
            if not p.exists() or not x: continue
            F = pickle.load(open(p, "rb"))
            for S, e, ci in x[1]:
                if ci is not None: rows.append((a, e["seed"], F, S, max(ci - 1, 0), x[1].index((S, e, ci)))); break
        if not rows: continue
        fig, ax = plt.subplots(2 * len(rows), 6, figsize=(18, 6 * len(rows)), squeeze=False)
        for r_, (a, sd, F, S, tc, epi) in enumerate(rows):
            ts = [t_ for t_ in range(max(tc - 25, 0), tc + 12) if (epi, t_) in F and t_ % 5 == 0][:6]
            for c, t_ in enumerate(ts): w, f = F[(epi, t_)]; ax[2 * r_, c].imshow(w); ax[2 * r_, c].set_title(f"{a} seed {sd} wrist step {t_ - tc:+d}", fontsize=8); ax[2 * r_ + 1, c].imshow(f); ax[2 * r_ + 1, c].set_title("front", fontsize=8)
            for c in range(6): ax[2 * r_, c].axis("off"); ax[2 * r_ + 1, c].axis("off")
        fig.tight_layout(); fig.savefig(OUT / f"g28_montage_{t}.png", dpi=40); plt.close(fig)
    pickle.dump({k: [{kk: vv for kk, vv in x.items() if not kk.startswith("curve")} for x in v] for k, v in R.items()}, open(OUT / "g28_summary.pkl", "wb"))


if __name__ == "__main__":
    main()
