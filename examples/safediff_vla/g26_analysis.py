#!/usr/bin/env python
"""G26 analysis. A: fruit orientation-freeze causal test (freeze0 baseline replicate and the G25 trace vs freeze1; seeds 3000-3004). B: toy GEO successful-vs-failed grasps at close/grasp (G25 toy seeds 3000-3004 + extra seeds 3005-3014, no intervention) against successful expert grasps. 'held' = is_grasped for >=30 post-close steps, 'lost' = < 30."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from g25_analysis import events, window_feats  # noqa: E402

O25 = Path("outputs/eval/safediff_vla_g25"); O26 = Path("outputs/eval/safediff_vla_g26")


def load_geo(p, task=None):
    L = pickle.load(open(p, "rb")); out = []
    for i, e in enumerate(L["episodes"]):
        S = [s for s in L["steps"] if s["ep"] == i]; ci = e["close"]
        if ci is None or not S: continue
        out.append((S, events(S, max(ci - 1, 0)), e))
    return out


def fmt(x): return "-" if x is None else f"{x:.0f}"


def main():
    print("================ A. FRUIT: orientation freeze (same GEO grasp / xyz / latched gripper; only EE orientation frozen at the grasp-confirmed pose) ================")
    ARMS = {"baseline G25 trace": load_geo(O25 / "select_fruit" / "g25_trace.pkl"), "baseline replicate (freeze0)": load_geo(O26 / "freeze0" / "select_fruit" / "g26_trace.pkl"), "FREEZE ROT (freeze1)": load_geo(O26 / "freeze1" / "select_fruit" / "g26_trace.pkl")}
    for nm, eps in ARMS.items():
        print(f"\n -- {nm} --")
        rows = []
        for S, ev, e in eps:
            tg = ev["t_grasp"]; post = window_feats(S, tg + 1, (ev["t_drop"] if ev["t_drop"] is not None else len(S))) if tg is not None else None; tot = min([np.linalg.norm(s["obj_p"][:2] - s["cont_p"][:2]) for s in S[ev["t_close"]:]]) * 100
            d = lambda k: (ev[k] - ev["t_close"]) if ev[k] is not None else None
            print(f"   seed {e['seed']}: success {int(e['success'])} len {e['len']} | grasp {fmt(d('t_grasp'))} lift {fmt(d('t_lift'))} slip {fmt(d('t_slip'))} drop {fmt(d('t_drop'))} goal_near {fmt(d('t_goal_near'))} held {ev['held_steps']} | min obj-goal xy {tot:.1f}cm" + (f" | transport: contacts {post['ncon']:.1f}, rel motion {post['rel_speed_cm']:.2f} cm/step, EE ang {post['ee_ang']:.4f} rad/step, width {post['width_mm']:.0f}mm" if post else ""))
            rows.append((e, ev, post, tot))
        n = len(rows); drops = sum(ev["t_drop"] is not None for _, ev, _, _ in rows); held = [ev["held_steps"] for _, ev, _, _ in rows]; post_ok = [p for _, _, p, _ in rows if p]
        print(f"   SUMMARY: success {sum(int(e['success']) for e, _, _, _ in rows)}/{n} | drops {drops}/{n} | held steps median {np.median(held):.0f} (mean {np.mean(held):.0f}) | held>=30: {sum(h >= 30 for h in held)}/{n} | mean min obj-goal xy {np.mean([t for *_, t in rows]):.1f}cm | transport EE ang {np.mean([p['ee_ang'] for p in post_ok]) if post_ok else float('nan'):.4f} rad/step, rel motion {np.mean([p['rel_speed_cm'] for p in post_ok]) if post_ok else float('nan'):.2f} cm/step, contacts {np.mean([p['ncon'] for p in post_ok]) if post_ok else float('nan'):.1f}")
    print("\n================ B. TOY: successful vs failed GEO grasps at close / grasp (no intervention) vs successful expert grasps ================")
    geo = load_geo(O25 / "select_toy" / "g25_trace.pkl") + (load_geo(O26 / "toy_extra" / "select_toy" / "g26_trace.pkl") if (O26 / "toy_extra" / "select_toy" / "g26_trace.pkl").exists() else [])
    EXP = []
    for e in pickle.load(open(O25 / "select_toy" / "expert_trace.pkl", "rb")):
        S = e["steps"]; tc = next((s["t"] for s in S if s["grip_cmd"] < 0.03), None)
        if tc is not None and e["success"]: EXP.append((S, events(S, tc), e))
    def feats(S, ev):
        tc = ev["t_close"]; s0 = S[tc]; n = len(S); Re = R.from_quat([s0["ee_q"][1], s0["ee_q"][2], s0["ee_q"][3], s0["ee_q"][0]]); zax = Re.apply([0, 0, 1]); tilt = float(np.degrees(np.arccos(np.clip(-zax[2], -1, 1)))); yaw = float(np.degrees(Re.as_euler("xyz")[2]))
        w10 = [S[min(tc + k, n - 1)] for k in range(1, 11)]; trel = np.array([s["t_rel"] for s in w10]) * 100
        return {"t_rel_close_cm": s0["t_rel"] * 100, "ee_above_obj_cm": (s0["ee_p"][2] - s0["obj_p"][2]) * 100, "ee_obj_xy_cm": float(np.linalg.norm(s0["ee_p"][:2] - s0["obj_p"][:2])) * 100, "tilt_deg": tilt, "yaw_deg": yaw, "rrel_deg": float(np.degrees(np.linalg.norm(s0["r_rel"]))), "width_close_mm": s0["width"] * 1000, "width_p3": S[min(tc + 3, n - 1)]["width"] * 1000, "width_p10": S[min(tc + 10, n - 1)]["width"] * 1000, "ncon_p1_10": np.mean([s["ncon"] for s in w10]), "trel10_move_cm": float(np.linalg.norm(trel[-1] - trel[0])), "trel10_final_cm": trel[-1]}
    print(f"   n GEO episodes {len(geo)} | held (>=30 steps) {sum(ev['held_steps'] >= 30 for _, ev, _ in geo)} | lost {sum(ev['held_steps'] < 30 for _, ev, _ in geo)} | expert successful grasps {len(EXP)}")
    keys = ["ee_above_obj_cm", "ee_obj_xy_cm", "tilt_deg", "rrel_deg", "width_close_mm", "width_p3", "width_p10", "ncon_p1_10", "trel10_move_cm"]
    print("\n   per GEO episode (seed, toy, held steps, success | EE z above object centre cm, EE-object XY cm, EE tilt from vertical deg, |object rotation in EE frame| deg, width at close / +3 / +10 mm, mean contacts 1-10, T_EE_object motion over first 10 steps cm, T_EE_object translation at +10 (x,y,z) cm)")
    rows = []
    for S, ev, e in geo:
        f = feats(S, ev); rows.append((f, ev, e)); print(f"   seed {e['seed']} {getattr(e, 'get', lambda *_: None)('target') or ''} held {ev['held_steps']:3d} succ {int(e['success'])} | {f['ee_above_obj_cm']:5.1f} {f['ee_obj_xy_cm']:5.1f} {f['tilt_deg']:5.1f} {f['rrel_deg']:6.1f} | {f['width_close_mm']:4.0f} {f['width_p3']:4.0f} {f['width_p10']:4.0f} | {f['ncon_p1_10']:.1f} | {f['trel10_move_cm']:5.1f} | ({f['trel10_final_cm'][0]:+.1f},{f['trel10_final_cm'][1]:+.1f},{f['trel10_final_cm'][2]:+.1f})")
    ef = [feats(S, ev) for S, ev, _ in EXP]
    print("\n   GROUP MEANS (sd):  " + "".join(f"{k:>18s}" for k in keys))
    def line(nm, fs): return f"   {nm:18s}(n={len(fs):2d}) " + "".join(f"{np.mean([x[k] for x in fs]):10.1f}({np.std([x[k] for x in fs]):4.1f})" for k in keys) if fs else f"   {nm}: none"
    print(line("EXPERT success", ef)); print(line("GEO held", [f for f, ev, e in rows if ev["held_steps"] >= 30])); print(line("GEO lost", [f for f, ev, e in rows if ev["held_steps"] < 30]))
    print("\n   separability of held vs lost (AUC, single feature, n=%d): " % len(rows) + ", ".join(f"{k} {_auc([f[k] for f, ev, e in rows], [ev['held_steps'] >= 30 for f, ev, e in rows]):.2f}" for k in keys))
    print("   lost GEO grasps vs the expert distribution (z of GEO-lost mean, in expert sd): " + ", ".join(f"{k} {(np.mean([f[k] for f, ev, e in rows if ev['held_steps'] < 30]) - np.mean([x[k] for x in ef])) / (np.std([x[k] for x in ef]) + 1e-9):+.1f}" for k in keys if any(ev['held_steps'] < 30 for _, ev, _ in rows)))


def _auc(x, y):
    x = np.array(x, float); y = np.array(y, bool)
    if y.all() or (~y).all(): return float("nan")
    a = x[y]; b = x[~y]; return float(np.mean([(p > q) + 0.5 * (p == q) for p in a for q in b]))


if __name__ == "__main__":
    main()
