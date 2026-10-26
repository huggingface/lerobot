#!/usr/bin/env python
"""G18-A analysis: relative held-out BC-loss increase when target-related conditioning is removed / shuffled, by horizon bin, stratum, task; counterfactual sensitivity; ridge probe."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np

OUT = Path("outputs/eval/safediff_vla_g18"); TASKS = ["poker", "mahjong", "fruit", "drink", "toy", "book"]; STR = ["EARLY", "MID", "NEAR", "POST"]; HBN = ["1-5", "6-10", "11-20", "21-30", "31-50"]


def loss(S, C, c, comp, t=None, st=None, hb=None):
    ti = slice(None) if t is None else [t]; si = slice(None) if st is None else [st]; hi = slice(None) if hb is None else [hb]
    n = C[c][ti][:, si][:, :, hi].sum(); return S[c][comp][ti][:, si][:, :, hi].sum() / n if n else np.nan


def rel(S, C, c, comp, base="N0", **k): b = loss(S, C, base, comp, **k); return (loss(S, C, c, comp, **k) - b) / b * 100 if b else np.nan


def main():
    L = pickle.load(open(OUT / "g18_loss.pkl", "rb")); S, C, sens = L["S"], L["CNT"], L["sens"]; n0 = C["N0"].sum(axis=(1, 2)); print(f"held-out frames' valid action steps per task: " + " ".join(f"{TASKS[i]}={int(n0[i] / 50)}" for i in range(6)) + f" | per stratum (frames) " + " ".join(f"{STR[j]}={int(C['N0'].sum(axis=(0, 2))[j] / 50)}" for j in range(4)))
    print("\n== ABSOLUTE held-out loss (encoded-space MSE) N0 normal, ALL tasks / frames, by horizon bin ==\n   comp      " + "".join(f"{h:>9s}" for h in HBN) + "     all")
    for comp in ("pos", "xy", "z", "rot", "grip"): print(f"   {comp:8s}  " + "".join(f"{loss(S, C, 'N0', comp, hb=i):9.4f}" for i in range(5)) + f"{loss(S, C, 'N0', comp):9.4f}")
    for scope, kw in (("ALL tasks", {}), ("POKER", {"t": 0}), ("FRUIT", {"t": 2}), ("mahjong", {"t": 1}), ("drink", {"t": 3}), ("toy", {"t": 4}), ("book", {"t": 5})):
        print(f"\n== RELATIVE increase vs N0 (%), {scope}: rows condition/comp, columns horizon bins 1-5 | 6-10 | 11-20 | 21-30 | 31-50 | all ==")
        for c in ("N1", "N2", "N4", "N1b"):
            base = "N0m" if c == "N2" else "N0"
            for comp in ("xy", "z", "pos", "rot", "grip"):
                if scope != "ALL tasks" and comp in ("rot",) and c != "N1": continue
                print(f"   {c:4s} {comp:5s}" + "".join(f"{rel(S, C, c, comp, base, hb=i, **kw):9.1f}" for i in range(5)) + f"{rel(S, C, c, comp, base, **kw):9.1f}")
    print("\n== RELATIVE increase of POSITION loss (%) by stratum x horizon (ALL tasks): N1 | N2 | N4 ==")
    for j, sn in enumerate(STR):
        print(f"   {sn:6s} " + " | ".join(f"{c}: " + " ".join(f"{rel(S, C, c, 'pos', 'N0m' if c == 'N2' else 'N0', st=j, hb=i):6.1f}" for i in range(5)) for c in ("N1", "N2", "N4")))
    print("   (columns = horizon bins 1-5 6-10 11-20 21-30 31-50)")
    # sensitivity
    print("\n== COUNTERFACTUAL SENSITIVITY: N0 chunk vs N2 chunk (target of another episode of the same task), cm; anchor separation = XY distance between the two predicted targets ==")
    print(f"   {'scope':10s}{'n':>6s}{'anchSep':>9s}{'sep10':>8s}{'sepEnd':>8s}{'sep10/anc':>10s}{'cos0->T':>9s}{'cos2->T':>9s}{'cos2->T2':>9s}{'off0':>7s}{'off2':>7s}")
    for nm, f in [("ALL", lambda r: True)] + [(TASKS[t], (lambda r, t=t: r["task"] == t)) for t in range(6)] + [(s, (lambda r, j=j: r["st"] == j)) for j, s in enumerate(STR)]:
        rr = [r for r in sens if f(r)]
        if rr: g = lambda k: np.mean([r[k] for r in rr]); print(f"   {nm:10s}{len(rr):6d}{g('anc_sep') * 100:9.1f}{g('sep10') * 100:8.2f}{g('sepEnd') * 100:8.2f}{g('sep10') / g('anc_sep'):10.2f}{g('cos0_true'):9.2f}{g('cos2_true'):9.2f}{g('cos2_used'):9.2f}{g('off0') * 100:7.2f}{g('off2') * 100:7.2f}")
    P = pickle.load(open(OUT / "g18_probe.pkl", "rb")); print(f"\n== TARGET-PREDICTABILITY PROBE (ridge, diagnostic; PRE_GRASP frames; train {P['n'][0]} / held-out-episode test {P['n'][1]}): R^2 for anchor x / y / z ==")
    names = []
    for k in P["P"]:
        if k[0] not in names and not k[0].startswith("anchor"): names.append(k[0])
    scopes = ["pooled", "within poker", "fit-in-task poker", "within fruit", "fit-in-task fruit"]
    print(f"   {'input':45s}" + "".join(f"{s:>26s}" for s in scopes))
    for n in names: print(f"   {n:45s}" + "".join((f"{'/'.join(f'{v:5.2f}' for v in P['P'][(n, s)]):>26s}" if (n, s) in P["P"] else f"{'-':>26s}") for s in scopes))
    print("   anchor spread (cm, std x/y/z) in held-out test: poker " + "/".join(f"{v:.1f}" for v in P["P"][("anchor std (cm) x/y/z", "poker")]) + " | fruit " + "/".join(f"{v:.1f}" for v in P["P"][("anchor std (cm) x/y/z", "fruit")]))


if __name__ == "__main__":
    main()
