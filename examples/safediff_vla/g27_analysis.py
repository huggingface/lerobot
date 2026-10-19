#!/usr/bin/env python
"""G27 analysis: toy grasp acquisition under three approaches (same oracle point, same close rule, same gripper, same POST_GRASP): A TOP-DOWN GEO (G25 seeds 3000-3004 + G26 toy_extra 3005-3014, decoder orientation) | B1 EXPERT-POSE (toy expert primitive orientation euler(-pi,0,0), approach along the gripper forward vector) | B2 SIDE (face-forward-horizontal euler(-pi/2,-pi/2,0), approach +y). Per episode: object between fingers after the close (width at +3 > 10mm), contacts (mean over steps 1-10; >=2 used as the bilateral proxy), grasped (is_grasped) held for the first 10 / 30 steps after grasp, lift (>2cm), native success."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g25_analysis import events  # noqa: E402

O25 = Path("outputs/eval/safediff_vla_g25"); O26 = Path("outputs/eval/safediff_vla_g26"); O27 = Path("outputs/eval/safediff_vla_g27")
TOY = {3000: "mickey", 3001: "pluto", 3002: "vision", 3003: "ace", 3004: "nami", 3005: "buzz_lightyear", 3006: "pluto", 3007: "jessie", 3008: "vision", 3009: "flashman", 3010: "nami", 3011: "buzz_lightyear", 3012: "sanji", 3013: "loki", 3014: "jessie"}
HELD_A = {"mickey", "pluto", "flashman", "loki"}


def load(p, seeds=None):
    L = pickle.load(open(p, "rb")); out = {}
    for i, e in enumerate(L["episodes"]):
        S = [s for s in L["steps"] if s["ep"] == i]; ci = e["close"]
        if ci is None or not S: out[e["seed"]] = None; continue
        out[e["seed"]] = (S, events(S, max(ci - 1, 0)), e)
    return out, L


def metrics(item):
    if item is None: return None
    S, ev, e = item; tc = ev["t_close"]; n = len(S); g = np.array([s["grasped"] for s in S]); tg = ev["t_grasp"]; w3 = S[min(tc + 3, n - 1)]["width"] * 1000; nc = np.mean([S[min(tc + k, n - 1)]["ncon"] for k in range(1, 11)])
    ret = lambda k: bool(tg is not None and tg + k < n and g[tg:tg + k].all()); lift = bool(ev["t_lift"] is not None)
    return {"between": w3 > 10, "w3": w3, "ncon": nc, "bilat": nc >= 2, "grasped": tg is not None and g[tc:].any(), "ret10": ret(10), "ret30": ret(30), "lift": lift, "held": ev["held_steps"], "success": bool(e["success"])}


def main():
    A = {}; LA = {}
    for p in (O25 / "select_toy" / "g25_trace.pkl", O26 / "toy_extra" / "select_toy" / "g26_trace.pkl"):
        d, L = load(p); A.update(d)
    arms = {"A TOP-DOWN GEO": A}
    for nm, d in (("B1 EXPERT-POSE", O27 / "expert_pose"), ("B2 SIDE", O27 / "side")):
        p = d / "select_toy" / "g27_trace.pkl"
        if p.exists(): arms[nm], Lb = load(p); print(f"{nm}: approach direction (gripper forward vector) = {np.round(Lb.get('pose_d'), 2) if Lb.get('pose_d') is not None else None}")
    print("\n== per seed (toy): between-fingers@+3 | contacts(1-10) | grasped | held first10/30 | lift | success ==")
    print(f"   {'seed':5s}{'toy':16s}" + "".join(f"{a[:2] + ' ' + a.split()[1]:>34s}" for a in arms))
    for sd in range(3000, 3015):
        line = f"   {sd:<5d}{TOY[sd]:16s}"
        for nm, d in arms.items():
            m = metrics(d.get(sd)); line += f"{('no close' if m is None else f'{int(m['between'])} w{m['w3']:4.0f} c{m['ncon']:.1f} g{int(m['grasped'])} {int(m['ret10'])}/{int(m['ret30'])} L{int(m['lift'])} S{int(m['success'])}'):>34s}"
        print(line)
    print("\n== SUMMARY (15 seeds; the 6 toys that were ALWAYS lost under A: vision, buzz_lightyear, jessie, nami, ace, sanji = 9 episodes | the 4 toys always held under A: mickey, pluto, flashman, loki = 6 episodes) ==")
    print(f"   {'arm':18s}{'group':14s}{'n':>3s}{'object between fingers':>24s}{'bilateral(>=2 contacts)':>24s}{'grasped':>9s}{'held 10':>8s}{'held 30':>8s}{'lift':>6s}{'success':>8s}")
    for nm, d in arms.items():
        for gn, cond in (("A-lost toys", lambda s: TOY[s] not in HELD_A), ("A-held toys", lambda s: TOY[s] in HELD_A), ("ALL", lambda s: True)):
            ms = [metrics(d.get(s)) for s in range(3000, 3015) if cond(s)]; n = len(ms); ms = [m for m in ms if m]
            f = lambda k: sum(bool(m[k]) for m in ms)
            print(f"   {nm:18s}{gn:14s}{n:3d}{f('between'):>21d}/{n}{f('bilat'):>21d}/{n}{f('grasped'):>6d}/{n}{f('ret10'):>5d}/{n}{f('ret30'):>5d}/{n}{f('lift'):>3d}/{n}{f('success'):>5d}/{n}")


if __name__ == "__main__":
    main()
