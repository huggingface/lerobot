#!/usr/bin/env python
"""G25 analysis: grasp-retention / transport audit for the latched ORACLE+GEOMETRY arm (toy, fruit; mahjong reference; seeds 3000-3004) against VLABench scripted-expert runs (same snapshot). Event times are step indices relative to t_close (GEO: the reactive-close step; expert: first finger command < 0.03).
t_grasp = first is_grasped step after t_close | t_lift = obj z > z(t_close)+2cm | t_drop = first step after t_grasp from which is_grasped is False for >=5 consecutive steps | t_slip = first step in [t_grasp+5, t_drop] where the object-in-EE transform moved >1cm / 0.1rad from its value at t_grasp+5 | t_goal_near = obj-goal XY < 15cm."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

OUT = Path("outputs/eval/safediff_vla_g25"); TASKS = ["select_toy", "select_fruit", "select_mahjong"]


def events(S, t_close):
    n = len(S); g = np.array([s["grasped"] for s in S]); oz = np.array([s["obj_p"][2] for s in S]); tg = next((t for t in range(t_close, n) if g[t]), None); ev = {"t_close": t_close, "t_grasp": tg}
    ev["t_lift"] = next((t for t in range(t_close, n) if oz[t] > oz[t_close] + 0.02), None); ev["t_drop"] = None; ev["t_slip"] = None
    if tg is not None:
        for t in range(tg, n - 4):
            if not g[t:t + 5].any(): ev["t_drop"] = t; break
        ref = min(tg + 5, n - 1); end = ev["t_drop"] if ev["t_drop"] is not None else n
        for t in range(ref, end):
            if np.linalg.norm(S[t]["t_rel"] - S[ref]["t_rel"]) > 0.01 or np.linalg.norm(S[t]["r_rel"] - S[ref]["r_rel"]) > 0.1: ev["t_slip"] = t; break
    gn = [t for t, s in enumerate(S) if "cont_p" in s and t >= t_close and np.linalg.norm(s["obj_p"][:2] - s["cont_p"][:2]) < 0.15]; ev["t_goal_near"] = gn[0] if gn else None; ev["held_steps"] = int(g[t_close:].sum()); return ev


def ee_ang_speed(S, t):
    if t < 1: return 0.0
    q0, q1 = S[t - 1]["ee_q"], S[t]["ee_q"]; return float((R.from_quat([q0[1], q0[2], q0[3], q0[0]]).inv() * R.from_quat([q1[1], q1[2], q1[3], q1[0]])).magnitude())


def window_feats(S, a, b):
    ts = [t for t in range(max(a, 2), min(b, len(S)))]
    if not ts: return None
    acc = [np.linalg.norm(S[t]["ee_p"] - 2 * S[t - 1]["ee_p"] + S[t - 2]["ee_p"]) * 100 for t in ts]; rel = [np.linalg.norm(S[t]["t_rel"] - S[t - 1]["t_rel"]) * 100 for t in ts]
    return {"ee_acc_cm": np.mean(acc), "ee_ang": np.mean([ee_ang_speed(S, t) for t in ts]), "dwidth_mm": (S[ts[-1]]["width"] - S[ts[0]]["width"]) * 1000, "width_mm": np.mean([S[t]["width"] for t in ts]) * 1000, "ncon": np.mean([S[t]["ncon"] for t in ts]), "rel_speed_cm": np.mean(rel), "obj_speed": np.mean([np.linalg.norm(S[t]["obj_v"]) for t in ts])}


def main():
    EX = {}; GEO = {}
    for t in TASKS:
        p = OUT / t / "expert_trace.pkl"
        if p.exists():
            E = pickle.load(open(p, "rb")); eps = []
            for e in E:
                S = e["steps"]; tc = next((s["t"] for s in S if s["grip_cmd"] < 0.03), None)
                if tc is not None and e["success"]: eps.append((S, events(S, tc)))
            EX[t] = eps
        p = OUT / t / "g25_trace.pkl"
        if p.exists():
            L = pickle.load(open(p, "rb")); eps = []
            for i, e in enumerate(L["episodes"]):
                S = [s for s in L["steps"] if s["ep"] == i]; ci = e["close"]
                if ci is None or not S: continue
                eps.append((S, events(S, max(ci - 1, 0)), e))
            GEO[t] = eps
    print("== EVENT TIMELINE (steps relative to t_close; '-' = never) ==")
    for t in TASKS:
        print(f"\n -- {t[7:]} --")
        for S, ev in EX.get(t, [])[:10]: print(f"   EXPERT n={len(S)}: grasp {ev['t_grasp'] - ev['t_close'] if ev['t_grasp'] is not None else '-'} lift {ev['t_lift'] - ev['t_close'] if ev['t_lift'] is not None else '-'} slip {ev['t_slip'] - ev['t_close'] if ev['t_slip'] is not None else '-'} drop {ev['t_drop'] - ev['t_close'] if ev['t_drop'] is not None else '-'} goal_near {ev['t_goal_near'] - ev['t_close'] if ev['t_goal_near'] is not None else '-'} held steps {ev['held_steps']}")
        for S, ev, e in GEO.get(t, []): f = lambda k: (ev[k] - ev["t_close"]) if ev[k] is not None else "-"; print(f"   GEO seed {e['seed']} success {int(e['success'])} len {e['len']}: grasp {f('t_grasp')} lift {f('t_lift')} slip {f('t_slip')} drop {f('t_drop')} goal_near {f('t_goal_near')} held steps {ev['held_steps']}")
    print("\n== STEP 2: object-in-gripper transform T_EE_object (EE frame; translation cm / rotation deg) at t_grasp+{5,10,20}: mean +- std over episodes; EXPERT vs GEO(all) vs GEO(held >=30 steps) vs GEO(held <30) ==")
    for t in TASKS:
        print(f" -- {t[7:]} --")
        for k in (5, 10, 20):
            def collect(eps):
                T, Rr = [], []
                for item in eps:
                    S, ev = item[0], item[1]
                    if ev["t_grasp"] is None or ev["t_grasp"] + k >= len(S): continue
                    s = S[ev["t_grasp"] + k]; T.append(s["t_rel"] * 100); Rr.append(np.degrees(s["r_rel"]))
                return np.array(T), np.array(Rr)
            geo_all = GEO.get(t, []); geo_hold = [x for x in geo_all if x[1]["held_steps"] >= 30]; geo_bad = [x for x in geo_all if x[1]["held_steps"] < 30]; line = f"   +{k:2d}:"
            for nm, eps in (("EXPERT", EX.get(t, [])), ("GEO all", geo_all), ("GEO held", geo_hold), ("GEO lost", geo_bad)):
                T, Rr = collect(eps)
                line += f" | {nm} n={len(T)}" + (f" t=({T.mean(0)[0]:+.1f},{T.mean(0)[1]:+.1f},{T.mean(0)[2]:+.1f}) sd {np.linalg.norm(T.std(0)):.1f} rot sd {np.linalg.norm(Rr.std(0)):.0f}" if len(T) else "")
            print(line)
    print("\n== width / contact at t_grasp+5 and early stability (first 20 steps after the grasp): finger width mm, contact count, object speed ==")
    for t in TASKS:
        for nm, eps in (("EXPERT", EX.get(t, [])), ("GEO", GEO.get(t, []))):
            w = [window_feats(item[0], item[1]["t_grasp"] + 3, item[1]["t_grasp"] + 20) for item in eps if item[1]["t_grasp"] is not None]; w = [x for x in w if x]
            if w: print(f"   {t[7:]:8s}{nm:8s} n={len(w)} width {np.mean([x['width_mm'] for x in w]):.1f}mm contact {np.mean([x['ncon'] for x in w]):.1f} rel-motion {np.mean([x['rel_speed_cm'] for x in w]):.2f} cm/step obj speed {np.mean([x['obj_speed'] for x in w]):.3f}")
    print("\n== STEP 3: slip predictors — window of 20 steps before the drop (GEO) vs expert transport window (t_lift .. t_lift+40) ==")
    print(f"   {'':22s}{'ee acc cm/step^2':>17s}{'ee ang rad/step':>16s}{'dwidth mm':>10s}{'width mm':>9s}{'contacts':>9s}{'rel motion cm/step':>19s}")
    for t in ("select_toy", "select_fruit"):
        rows = []
        for item in GEO.get(t, []):
            S, ev = item[0], item[1]
            if ev["t_drop"] is not None: w = window_feats(S, ev["t_drop"] - 20, ev["t_drop"])
            else: w = None
            if w: rows.append(w)
        ex = [window_feats(S, ev["t_lift"], ev["t_lift"] + 40) for S, ev in EX.get(t, []) if ev["t_lift"] is not None]; ex = [x for x in ex if x]
        for nm, ww in ((f"{t[7:]} GEO pre-drop", rows), (f"{t[7:]} EXPERT transport", ex)):
            if ww: print(f"   {nm:22s}{np.mean([x['ee_acc_cm'] for x in ww]):17.2f}{np.mean([x['ee_ang'] for x in ww]):16.3f}{np.mean([x['dwidth_mm'] for x in ww]):10.2f}{np.mean([x['width_mm'] for x in ww]):9.1f}{np.mean([x['ncon'] for x in ww]):9.1f}{np.mean([x['rel_speed_cm'] for x in ww]):19.2f}   (n={len(ww)})")
    print("\n== GEO executed gripper command after the close (hold command interface): fraction of post-close steps with command CLOSED, and finger width trend ==")
    for t in TASKS:
        for S, ev, e in GEO.get(t, []):
            cg = np.array([c if c is not None else np.nan for c in e["cmd_grip"]]); tc = ev["t_close"]; w = np.array([s["width"] for s in S])
            print(f"   {t[7:]:8s} seed {e['seed']}: closed-cmd frac {np.mean(cg[tc + 1:] <= 0.5):.2f} | width at close {w[tc] * 1000:.0f}mm, +10 {w[min(tc + 10, len(w) - 1)] * 1000:.0f}mm, +30 {w[min(tc + 30, len(w) - 1)] * 1000:.0f}mm, end {w[-1] * 1000:.0f}mm | held steps {ev['held_steps']}")
    pickle.dump({"EX": {k: [(e[1]) for e in v] for k, v in EX.items()}}, open(OUT / "g25_summary.pkl", "wb"))


if __name__ == "__main__":
    main()
