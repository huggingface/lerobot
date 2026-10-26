#!/usr/bin/env python
"""G17 analysis (all numbers from stored fixed-observation chunks; no training). Conditions: V2 | G4 (gamma 4, upper bound) | NT (no target) | move_I1..I4 (target vector moved to a later location, input branch removed) | extra_I0..I4 (V2 + one extra copy at the location) | geo_I0..I4 (V2 + trained G16 h_geo at the location)."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g17_capture import PTS  # noqa: E402
from g16_analysis import stratum  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g17"); TASK = ["poker", "mahjong", "fruit", "drink", "toy", "book"]


def sep(F, c, sl, a="GT", b="WR"): return np.mean([np.linalg.norm((f["cond"][(c, a)]["xyz"][sl, :2] - f["cond"][(c, b)]["xyz"][sl, :2]).mean(0)) for f in F]) * 100


def cosv(f, c, a="GT"):
    anc = f["gt"] if a == "GT" else f["wr"]; v = f["cond"][(c, a)]["xyz"][:10, :2].mean(0) - f["ee"][:2]; d = anc[:2] - f["ee"][:2]; return v @ d / (np.linalg.norm(v) * np.linalg.norm(d) + 1e-9)


def cmet(F, c):
    x = lambda f: f["cond"][(c, "GT")]["xyz"]; q = lambda f: x(f)
    err = lambda f, k, sl: np.linalg.norm(x(f)[k, sl] - f["gt"][sl])
    jerk = np.mean([np.linalg.norm(np.diff(x(f), n=3, axis=0), axis=1).mean() for f in F]) * 100; ms = np.mean([np.linalg.norm(np.diff(x(f), axis=0), axis=1).max() for f in F]) * 100; fo = np.array([np.linalg.norm(x(f)[0] - f["ee"]) for f in F]) * 100
    return [sep(F, c, slice(0, 5)), sep(F, c, slice(0, 10)), sep(F, c, slice(49, 50)), np.mean([cosv(f, c) for f in F]), np.mean([cosv(f, c, "WR") for f in F]), np.mean([err(f, 29, slice(0, 2)) for f in F]) * 100, np.mean([err(f, 29, slice(2, 3)) for f in F]) * 100, np.mean([err(f, 29, slice(0, 3)) for f in F]) * 100, fo.mean(), np.mean(fo > 5), ms, jerk,
            np.mean([np.argmax(f["cond"][(c, "GT")]["grip"] <= .5) if (f["cond"][(c, "GT")]["grip"] <= .5).any() else 50 for f in F])]


COLS = ["sep5", "sep10", "sepEnd", "cos->GT", "cos->WR", "XY@30", "Z@30", "3D@30", "1stOff", ">5cm", "maxStp", "jerk", "closeIdx"]


def table(F, names, title):
    print(f"\n== {title} (n={len(F)}) ==\n   {'condition':12s}" + "".join(f"{c:>8s}" for c in COLS))
    for n in names: print(f"   {n:12s}" + "".join(f"{v:8.2f}" for v in cmet(F, n)))


def gain_bank(F, c, ax, sl):
    a, b = "xyz".index(ax), None; return np.mean([(f["cond"][(c, ax + "p")]["xyz"][sl, a] - f["cond"][(c, ax + "m")]["xyz"][sl, a]).mean() / 0.10 for f in F])


def main():
    D = pickle.load(open(OUT / "g17_bank.pkl", "rb")); F = D["R"]
    for f in F: f["stratum"] = stratum(f)
    print(f"frames {len(F)} | runner vs dec.forward max diff {D['maxdiff']:.1e}\nDECODER: nn.TransformerDecoder, 4 post-norm layers (LN after each residual), hidden 512, FFN 1024, 8 heads, no final norm; head = Linear(512->10) per horizon token. Target branch = additive term in the INPUT queries (I0). Mean ||target_xyz_encoder(T)|| = {np.mean([f['prop']['nt_in_norm'] for f in F]):.2f}; G16 h_geo mean norm {np.mean([f['prop']['hgeo_norm'] for f in F]):.2f}")
    # ---- A/B ----
    print("\n== G17-A/B: hidden difference between GT and wrong-card anchors along the decoder (normal V2 pipeline). d = mean_token ||dh||, rel = d/||h||, cos = cos(dh, input-level target-induced dh), ratio = d / d at the previous residual-stream point ==")
    pts_stream = ["in"] + [f"L{l}.{p}" for l in range(4) for p in ("x1", "x2", "x3")]
    for sname in ("ALL", "EARLY", "MID", "LATE", "NEAR/CONTACT"):
        S = [f for f in F if sname == "ALL" or f["stratum"] == sname]
        if not S: continue
        print(f" [{sname}] n={len(S)}   {'point':8s}{'d':>8s}{'rel':>8s}{'cos':>7s}{'ratio':>7s}" + ("   (sublayer outputs: SA / CA / FFN d per layer)" if sname == "ALL" else ""))
        prev = None
        for p in PTS:
            d = np.mean([f["prop"][p]["d"] for f in S]); h = np.mean([f["prop"][p]["h"] for f in S]); cs = np.nanmean([f["prop"][p]["cos"] for f in S]) if p != "out" else float("nan"); r = d / prev if (prev and p in pts_stream) else float("nan")
            if p in pts_stream or p == "out": print(f"          {p:8s}{d:8.3f}{d / h:8.3f}{cs:7.2f}{r:7.2f}"); prev = d if p in pts_stream else prev
        if sname == "ALL":
            for l in range(4): print(f"          layer{l} sublayer dh: SA {np.mean([f['prop'][f'L{l}.sa']['d'] for f in S]):.3f} | CA {np.mean([f['prop'][f'L{l}.ca']['d'] for f in S]):.3f} | FFN {np.mean([f['prop'][f'L{l}.ff']['d'] for f in S]):.3f}   (resid stream before LN has |dh| = after previous LN d; LN output d above)")
    # ---- C/D ----
    names = ["NT", "V2", "G4"] + [f"move_I{k}" for k in range(1, 5)] + [f"extra_I{k}" for k in range(5)] + [f"geo_I{k}" for k in range(5)]
    for sname, S in (("ALL frames", F), ("PRE_GRASP-open", [f for f in F if f["phase"] == 0 and f["grip_raw"] <= .5])):
        table(S, names, f"G17-C/D counterfactual GT vs wrong-card, {sname}. sep* = XY separation (cm; anchor separation = {np.mean([np.linalg.norm(f['gt'][:2] - f['wr'][:2]) for f in S]) * 100:.1f}); errors cm to GT anchor; V2 == move_I0")
    S = F; print("\n== G4d predicted-anchor secondary (ALL frames): 3D error @30 / first-offset ==")
    for n in names:
        e = np.mean([np.linalg.norm(f["cond"][(n, "G4D")]["xyz"][29] - f["gt"]) for f in S]) * 100; fo = np.mean([np.linalg.norm(f["cond"][(n, "G4D")]["xyz"][0] - f["ee"]) for f in S]) * 100; print(f"   {n:12s} {e:7.2f} {fo:6.2f}")
    # ---- E bank ----
    print("\n== G17-E (bank, poker): anchor gain (d chunk / d anchor, +-5cm; 1 = one-for-one) first10 | late(30:50). z is OUT-OF-DISTRIBUTION for poker (secondary; the 6-task dataset Z analysis follows) ==")
    print(f"   {'condition':12s}" + "".join(f"{a + ' ' + s:>11s}" for a in "xyz" for s in ("first10", "late")))
    for n in names[1:]: print(f"   {n:12s}" + "".join(f"{gain_bank(F, n, a, sl):11.3f}" for a in "xyz" for sl in (slice(0, 10), slice(30, 50))))
    # ---- E dataset ----
    DE = pickle.load(open(OUT / "g17_dataset_E.pkl", "rb")); rows = DE["rows"]; zs = {}
    for t, z in DE["zstat"]: zs.setdefault(t, []).append(z)
    print("\n== G17-E (6-task dataset, held-out episodes, PRE_GRASP, demonstration anchor): natural anchor-z spread per task (mean +- std, mm) ==\n   " + " | ".join(f"{t[7:]}: {np.mean(v) * 1000:.0f}+-{np.std(v) * 1000:.1f} (n={len(v)})" for t, v in zs.items()))
    print("   gain per task (first10 | late), x/y +-5cm, z +-3cm; Z reliable only where natural spread is non-trivial")
    keys = [("V2", None), ("G4", None)] + [("move", f"I{k}") for k in range(1, 5)] + [("extra", f"I{k}") for k in range(5)] + [("geo", f"I{k}") for k in range(5)]
    for ax in "xyz":
        print(f" axis {ax}:   {'condition':12s}" + "".join(f"{t:>14s}" for t in TASK) + "         ALL")
        for k in keys:
            rs = [r for r in rows if r["cond"] == k]; line = ""
            for t in TASK + [None]:
                v1 = np.concatenate([r["gain"][ax]["first10"][[x == "select_" + t if t else True for x in r["tasks"]]] for r in rs]); v2 = np.concatenate([r["gain"][ax]["late"][[x == "select_" + t if t else True for x in r["tasks"]]] for r in rs]); line += f"  {v1.mean():+5.2f}|{v2.mean():+5.2f}" if len(v1) else "       n/a   "
            print(f"   {k[0] + ('_' + k[1] if k[1] else ''):12s}" + line)
    # ---- F ----
    PF = pickle.load(open(OUT / "g17_dataset_F.pkl", "rb")); print(f"\n== G17-F ridge probes (diagnostic only): held-out-episode R^2 of residual r_k = action_k - EE; frames train/test {PF['n']} ==\n   {'k comp':8s}{'H (final hidden)':>18s}{'G (geometry)':>14s}{'H+G':>8s}")
    for k in (10, 30):
        for c in "xyz": print(f"   {k:3d} {c}    {PF['probe'][(k, c, 'H')]:18.3f}{PF['probe'][(k, c, 'G')]:14.3f}{PF['probe'][(k, c, 'H+G')]:8.3f}")


if __name__ == "__main__":
    main()
