#!/usr/bin/env python
"""G23 analysis. Part A: release ground truth (release_gt.txt). Part B: GEO post-grasp timelines (re-execution logs). Part C: decoder gripper output on expert post-grasp states vs GEO-created post-grasp states, stratified by goal proximity. Part D: release-driver stratification / logistic regression and interventions."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
OUT = Path("outputs/eval/safediff_vla_g23"); TASKS = ["select_poker", "select_mahjong", "select_toy", "select_fruit"]; CHUNK = 50


def load_consts():
    from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT
    from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
    p = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT); return float(p.action_grip_mean), float(p.action_grip_std)


def main():
    gm, gs = load_consts(); thr_n = (0.5 - gm) / gs; RG = pd.read_pickle(OUT / "release_gt.pkl"); df = pd.read_pickle(OUT / "g23_fixed.pkl")
    medz = {t: float(np.median(np.stack(RG[(RG.task == t) & RG.t_rel.notna()].ee_rel.values)[:, 2])) for t in TASKS if ((RG.task == t) & RG.t_rel.notna()).any()}
    print("== PART B: GEO post-grasp timelines (logging re-execution of the G21 geo arm; the executed gripper channel is the reactive-close latch: forced CLOSED from the close step to the episode end) ==")
    G = {}
    for t in ("select_toy", "select_mahjong", "select_fruit"):
        p = OUT / t / "g23_geo_log.pkl"
        if not p.exists(): continue
        L = pickle.load(open(p, "rb")); G[t] = L; print(f"\n -- {t[7:]} (success per seed {[int(e['success']) for e in L['episodes']]}; G21 geo success was: {'toy 0/5' if t.endswith('toy') else ('mahjong 4/5' if t.endswith('mahjong') else 'fruit 0/5')}) --")
        for i, e in enumerate(L["episodes"]):
            S = [s for s in L["steps"] if s["ep"] == i]; ci = e["close"]; cg = np.array([c if c is not None else np.nan for c in e["cmd_grip"]]); post = [r for r in L["replans"] if r["ep"] == i and r["phase"] == 1]
            cont = np.array([s["cont"] for s in S]); obj = np.array([s["obj"] for s in S]); ee = np.array([s["ee"] for s in S]); dxy_ee = np.linalg.norm(ee[:, :2] - cont[:, :2], axis=1) * 100; dxy_obj = np.linalg.norm(obj[:, :2] - cont[:, :2], axis=1) * 100
            first_near = next((s["t"] for s, d in zip(S, dxy_ee) if d < 10 and s["t"] > (ci or 0)), None); dm = [np.max(r["grip"]) for r in post]; anyopen = [(r["grip"] > 0.5).any() for r in post]
            print(f"   seed {e['seed']}: len {e['len']} close@{ci} | closed after close {np.mean(cg[ci:] <= 0.5) if ci else float('nan'):.2f}, closed->open transitions {int(np.sum((cg[1:] > 0.5) & (cg[:-1] <= 0.5)))} | min EE-goal xy {dxy_ee[ci:].min() if ci else float('nan'):.1f}cm (first <10cm at step {first_near}), min obj-goal xy {dxy_obj[ci:].min() if ci else float('nan'):.1f}cm, obj grasped steps {int(np.sum([s['grasped'] for s in S]))} | post-grasp decoder replans {len(post)}: chunks with an OPEN command {int(np.sum(anyopen))}/{len(post)}, max decoder grip {max(dm) if dm else float('nan'):.2f}")
    print("\n== PART C/D: decoder gripper chunk on EXPERT post-grasp states (held-out episodes; frozen demo anchor; phase=1) ==")
    ex = np.stack(df.expert_g.values) if False else None
    df["exp_open_any"] = [(np.array(x) > thr_n).any() for x in df.expert_g]; df["exp_first_open"] = [int(np.argmax(np.array(x) > thr_n)) if (np.array(x) > thr_n).any() else CHUNK for x in df.expert_g]
    for k in ("base", "phase0", "noimg"): df[f"{k}_any"] = [(np.array(x) > 0.5).any() for x in df[f"g_{k}"]]; df[f"{k}_first"] = [int(np.argmax(np.array(x) > 0.5)) if (np.array(x) > 0.5).any() else CHUNK for x in df[f"g_{k}"]]; df[f"{k}_frac"] = [(np.array(x) > 0.5).mean() for x in df[f"g_{k}"]]
    print(f"   {'task':14s}{'n':>6s}{'expert opens in chunk':>22s}{'dec opens (base)':>17s}{'P(dec|expert opens)':>20s}{'P(dec|expert not)':>19s}{'first-open err (steps, both open)':>34s}{'dec phase0':>11s}{'dec noimg':>10s}")
    for t in TASKS:
        d = df[df.task == t]; eo = d.exp_open_any.values; do = d.base_any.values; both = d[eo & do]
        print(f"   {t[7:]:14s}{len(d):6d}{eo.mean():22.2f}{do.mean():17.2f}{(do[eo].mean() if eo.any() else float('nan')):20.2f}{(do[~eo].mean() if (~eo).any() else float('nan')):19.2f}{((both.base_first - both.exp_first_open).abs().mean() if len(both) else float('nan')):34.1f}{d.phase0_any.mean():11.2f}{d.noimg_any.mean():10.2f}")
    print("\n   decoder opens (base) by the expert's remaining time to release (frames of episodes with a release; bins of t_rel - t):")
    print(f"   {'task':14s}" + "".join(f"{b:>12s}" for b in ("<=10", "11-25", "26-50", "51-100", ">100")))
    for t in TASKS:
        d = df[(df.task == t) & df.t_rel.notna()].copy(); d["rem"] = d.t_rel - d.t; line = ""
        for lo, hi in ((-1, 10), (10, 25), (25, 50), (50, 100), (100, 10 ** 6)): m = d[(d.rem > lo) & (d.rem <= hi)]; line += f"{(m.base_any.mean() if len(m) else float('nan')):7.2f}({len(m):3d})" if len(m) else f"{'-':>12s}"
        print(f"   {t[7:]:14s}{line}")
    print("\n   Step 4: logistic regression, y = decoder opens within the chunk (base), standardised features; AUC / coefficients (positive = more open)")
    for t in TASKS[1:]:
        d = df[(df.task == t) & df.t_rel.notna()].copy()
        if len(d) < 100: continue
        d["tsg"] = d.t - d.tc; X = d[["tsg", "d_rel", "ee_z"]].values; X = (X - X.mean(0)) / X.std(0); y = d.base_any.values.astype(int)
        if y.min() == y.max(): continue
        m = LogisticRegression(max_iter=500).fit(X, y); from sklearn.metrics import roc_auc_score; auc = roc_auc_score(y, m.predict_proba(X)[:, 1]); a1 = roc_auc_score(y, X[:, 0]); a2 = roc_auc_score(y, -X[:, 1]); a3 = roc_auc_score(y, -X[:, 2])
        print(f"   {t[7:]:10s} n={len(d)} | joint AUC {auc:.2f} | single-feature AUC: elapsed time since grasp {a1:.2f}, closeness to release point {a2:.2f}, low EE height {a3:.2f} | coef tsg/d_rel/z = {m.coef_[0][0]:+.2f}/{m.coef_[0][1]:+.2f}/{m.coef_[0][2]:+.2f}")
    print("\n   Part C: GEO-created post-grasp decoder replans vs expert states at matched goal proximity (EE distance to the goal point; GEO: container xy + the task's median demonstrated release height; expert: the demonstration's release point), % chunks with an OPEN command")
    for t in ("select_toy", "select_mahjong", "select_fruit"):
        if t not in G: continue
        L = G[t]; rows = []
        for r in L["replans"]:
            if r["phase"] != 1: continue
            S = [s for s in L["steps"] if s["ep"] == r["ep"] and s["t"] == r["t"]]
            if not S: continue
            s = S[0]; goal = np.array([s["cont"][0], s["cont"][1], medz[t]]); rows.append((float(np.linalg.norm(s["ee"] - goal)) * 100, bool((r["grip"] > 0.5).any())))
        gA = pd.DataFrame(rows, columns=["d", "open"]); d = df[(df.task == t) & df.t_rel.notna()].copy(); d["dcm"] = d.d_rel * 100; line = f"   {t[7:]:8s}"
        for lo, hi in ((0, 15), (15, 30), (30, 50), (50, 1e9)):
            a = gA[(gA.d >= lo) & (gA.d < hi)]; b = d[(d.dcm >= lo) & (d.dcm < hi)]; line += f" | {lo:.0f}-{hi if hi < 1e8 else 999:.0f}cm: GEO {a.open.mean() * 100 if len(a) else float('nan'):5.0f}% (n={len(a)}) vs expert {b.base_any.mean() * 100 if len(b) else float('nan'):5.0f}% (n={len(b)})"
        print(line)
    print("\n   Interventions on the same expert frames (decoder opens within chunk, %), base / phase flag forced 0 / image tokens masked:")
    for t in TASKS: d = df[df.task == t]; print(f"   {t[7:]:10s} {d.base_any.mean() * 100:5.0f} / {d.phase0_any.mean() * 100:5.0f} / {d.noimg_any.mean() * 100:5.0f}   (mean open fraction {d.base_frac.mean():.2f} / {d.phase0_frac.mean():.2f} / {d.noimg_frac.mean():.2f})")


if __name__ == "__main__":
    main()
