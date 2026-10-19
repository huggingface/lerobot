#!/usr/bin/env python
"""G30 analysis (DRINK, seeds 3000-3004): baseline = G28 Arm C (waypoint + pose, 3cm reactive close; reused) vs the same trajectory with close at 1cm grasp-pose completion. At the close: approach-axis residual and lateral offset (EE - keypoint), finger-object geometry (object centre projected on the finger-tip axis: parameter t in [0,1] = between the tips, and its perpendicular distance), width at +1/+3/+5, contact count, actual grasp (is_grasped), object displacement, native success."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from g28_analysis import ep_metrics  # noqa: E402
from g25_analysis import events  # noqa: E402

O28 = Path("outputs/eval/safediff_vla_g28"); O29 = Path("outputs/eval/safediff_vla_g30"); SEEDS = [3000, 3001, 3002, 3003, 3004]


def load_g28():
    L = pickle.load(open(O28 / "select_drink" / "C" / "g28_trace.pkl", "rb")); out = []
    for i, e in enumerate(L["episodes"]):
        S = [s for s in L["steps"] if s["ep"] == i]; ci = next((k for k, r in enumerate(e["rec"]) if r["reactive_close_event"]), None); out.append((S, e, ci, None))
    return out


def load_g29(arm):
    out = []
    for sd in SEEDS:
        p = O29 / "select_drink" / arm / f"seed{sd}" / "g28_trace.pkl"
        if not p.exists(): continue
        L = pickle.load(open(p, "rb")); e = L["episodes"][0]; S = [s for s in L["steps"] if s["ep"] == 0]; ci = next((k for k, r in enumerate(e["rec"]) if r["reactive_close_event"]), None); out.append((S, e, ci, L))
    return out


def geom(S, tc, k=0):
    s = S[min(tc + k, len(S) - 1)]
    if "ltip" not in s: return None
    a, b = s["ltip"], s["rtip"]; u = b - a; L = np.linalg.norm(u); u = u / L; o = s["obj_p"]; t = float(((o - a) @ u) / L); perp = float(np.linalg.norm((o - a) - ((o - a) @ u) * u)) * 100; return t, perp, L * 1000


def main():
    ARM = {"A (G28 Arm C, 3cm close; reused)": load_g28(), "B same trajectory, 1cm completion close": load_g29("C")}
    print(f"{'arm':44s}{'n':>3s}{'close fired':>12s}{'axial resid@close cm':>22s}{'lateral@close cm':>17s}{'between@+1':>11s}{'@+3':>5s}{'@+5':>5s}{'ncon(1-10)':>11s}{'is_grasped':>11s}{'tgt moved>1.5cm':>16s}{'lift':>5s}{'native':>7s}")
    R = {}
    for nm, eps in ARM.items():
        ms = []
        for S, e, ci, L in eps:
            m = ep_metrics(S, e, ci); m["L"] = L
            if m["close"]:
                tc = max(ci - 1, 0); n = len(S); m["w1"] = S[min(tc + 1, n - 1)]["width"] * 1000; m["w5"] = S[min(tc + 5, n - 1)]["width"] * 1000; m["ncon_close"] = S[tc]["ncon"]; m["grasped_any"] = any(s["grasped"] for s in S[tc:]); m["geom0"] = geom(S, tc, 0); m["S"] = S; m["tc"] = tc
            ms.append(m)
        R[nm] = ms; n = len(ms); cl = [m for m in ms if m["close"]]; f = lambda k: sum(bool(m[k]) for m in cl)
        print(f"{nm:44s}{n:3d}{len(cl):>9d}/{n}{(np.mean([m['axial_behind_cm'] for m in cl]) if cl else float('nan')):22.1f}{(np.mean([m['lat_close_cm'] for m in cl]) if cl else float('nan')):17.1f}{sum(m['w1'] > 10 for m in cl if 'w1' in m):>8d}/{len(cl)}{f('between'):>3d}/{len(cl)}{sum(m['w5'] > 10 for m in cl if 'w5' in m):>3d}/{len(cl)}{(np.mean([m['ncon'] for m in cl]) if cl else float('nan')):11.1f}{sum(m.get('grasped_any', False) for m in cl):>8d}/{len(cl)}{sum(m['post_move_cm'] > 1.5 for m in cl):>12d}/{len(cl)}{f('lift'):>3d}/{len(cl)}{sum(m['success'] for m in ms):>5d}/{n}")
    print("\nper episode: seed: min dist cm | axial residual / lateral at close cm | width +1/+3/+5 mm | contacts at close / mean(1-10) | finger geometry at close (t of object centre between tips [0..1], perpendicular cm, tip gap mm) | is_grasped | target displacement after close cm | native")
    for nm, ms in R.items():
        print(f"  {nm}")
        for m in ms:
            if not m["close"]: print(f"     {m['seed']}: min {m['min_d_cm']:.1f} NO CLOSE (pre-close target/other displacement {m['pre_t_cm']:.1f}/{m['pre_o_cm']:.1f} cm) native {int(m['success'])}"); continue
            g = m.get("geom0"); gs = f"t={g[0]:.2f} perp={g[1]:.1f} gap={g[2]:.0f}" if g else "n/a (not logged in G28)"
            print(f"     {m['seed']}: min {m['min_d_cm']:.1f} | {m['axial_behind_cm']:.1f}/{m['lat_close_cm']:.1f} | {m.get('w1', float('nan')):.0f}/{m['w3']:.0f}/{m.get('w5', float('nan')):.0f} | {m.get('ncon_close', '-')}/{m['ncon']:.1f} | {gs} | {int(m.get('grasped_any', False))} | {m['post_move_cm']:.1f} | {int(m['success'])}")
    # montage of close moment: wrist+front frames near the close for baseline (no frames for A reused from G28 pkl), B and C
    fig, ax = plt.subplots(8, 6, figsize=(18, 24)); r_ = 0
    keys = list(R)
    for nm, key, seed in (("A (3cm)", keys[0], 3000), ("B (1cm)", keys[1], 3000), ("A (3cm)", keys[0], 3002), ("B (1cm)", keys[1], 3002)):
        ms = R[key]; m = next((x for x in ms if x["seed"] == seed), None)
        if m is None: r_ += 2; continue
        if key == keys[0]: F = pickle.load(open(O28 / "select_drink" / "C" / "g28_frames.pkl", "rb")); ep = SEEDS.index(seed)
        else: F = pickle.load(open(O29 / "select_drink" / "C" / f"seed{seed}" / "g28_frames.pkl", "rb")); ep = 0
        tc = m["tc"] if m["close"] else 60; ts = [t for t in range(max(tc - 15, 0), tc + 25) if (ep, t) in F and t % 5 == 0][:6]
        for c, t in enumerate(ts): w, f = F[(ep, t)]; ax[r_, c].imshow(w); ax[r_, c].set_title(f"{nm} seed {seed} wrist {t - tc:+d}", fontsize=8); ax[r_ + 1, c].imshow(f); ax[r_ + 1, c].set_title("front", fontsize=8)
        r_ += 2
    for a_ in ax.ravel(): a_.axis("off")
    fig.tight_layout(); fig.savefig(O29 / "g30_close_montage.png", dpi=34); plt.close(fig)


if __name__ == "__main__":
    main()
