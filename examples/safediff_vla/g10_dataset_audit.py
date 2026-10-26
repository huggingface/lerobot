#!/usr/bin/env python
"""G10-Z Step 6 (+ Step 1 demo part): DATASET SUPERVISION AUDIT on lerobot/vlabench_unified (the data V2 / the decoder were trained on). Raw robot-frame metres. No model involved.
Questions: does the data give the decoder a reason to use target z? how tight is the vertical template (z vs time, z vs close-relative time, z vs XY distance), and how well do candidate predictors explain the demonstrated future z?"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g10_audit"); OUT.mkdir(parents=True, exist_ok=True)


def r2(y, X):
    X = np.c_[np.ones(len(y)), X]; b, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ b; return float(1 - res.var() / y.var()) if y.var() > 0 else float("nan")


def binned_r2(y, x, nb=20):
    q = np.quantile(x, np.linspace(0, 1, nb + 1)); idx = np.clip(np.searchsorted(q, x, side="right") - 1, 0, nb - 1); m = np.array([y[idx == i].mean() if (idx == i).any() else y.mean() for i in range(nb)]); return float(1 - (y - m[idx]).var() / y.var())


def main():
    d = LeRobotDataset("lerobot/vlabench_unified"); fi = d.meta.episodes["dataset_from_index"]; ti = d.meta.episodes["dataset_to_index"]; E = pd.read_json("outputs/eval/safediff_vla_phase_d0_training_data_audit/episodes.json"); res = {}
    print("== per-task target z / close statistics (500 episodes each) ==")
    for task, g in E.groupby("task"):
        an = np.array(g.anchor.tolist()); res[task] = {"anchor_z_mean": float(an[:, 2].mean()), "anchor_z_std": float(an[:, 2].std()), "t_close_mean": float(g.t_close.mean()), "t_close_std": float(g.t_close.std())}
        print(f"   {task:15s} anchor z mean {an[:, 2].mean():.4f} std {an[:, 2].std() * 1000:.2f}mm | t_close {g.t_close.mean():.1f} +- {g.t_close.std():.1f}")
    P = E[E.task == "select_poker"].reset_index(drop=True); eps = []
    for _, r in P.iterrows():
        ep = int(r.episode); sl = d.hf_dataset[int(fi[ep]):int(ti[ep])]; eps.append({"ep": ep, "a": np.array(sl["action"]), "s": np.array(sl["observation.state"]), "tc": int(r.t_close), "anchor": np.array(r.anchor), "ref": r.referent})
    print(f"\n== POKER demonstrations: {len(eps)} episodes, lengths {np.min([len(e['a']) for e in eps])}-{np.max([len(e['a']) for e in eps])} ==")
    zc = np.array([e["s"][e["tc"], 2] for e in eps]); azc = np.array([e["a"][e["tc"], 2] for e in eps]); z0 = np.array([e["s"][0, 2] for e in eps]); print(f"   EE z at start {z0.mean():.3f}+-{z0.std() * 1000:.1f}mm | EE z at close {zc.mean():.4f}+-{zc.std() * 1000:.2f}mm | action z at close {azc.mean():.4f}+-{azc.std() * 1000:.2f}mm | target z {np.mean([e['anchor'][2] for e in eps]):.4f}+-{np.std([e['anchor'][2] for e in eps]) * 1000:.2f}mm")
    print("   close-relative EE z (mean +- std across episodes, mm) at tau = t - t_close:")
    for tau in (-50, -40, -30, -20, -10, -5, 0, 5, 10):
        v = np.array([e["s"][e["tc"] + tau, 2] for e in eps if 0 <= e["tc"] + tau < len(e["s"])]); print(f"      tau {tau:+3d}: z {v.mean():.4f} +- {v.std() * 1000:6.2f}mm (n={len(v)})")
    print("   fixed-time EE z (mean +- std across episodes): " + " | ".join(f"t={t}: {np.mean([e['s'][t, 2] for e in eps]):.3f}+-{np.std([e['s'][t, 2] for e in eps]) * 1000:.0f}mm" for t in (0, 10, 20, 30, 40, 50, 60)))
    xyd = lambda e, t: np.linalg.norm(e["s"][t, :2] - e["anchor"][:2])
    print("   EE-target XY distance at the same fixed times (mean +- std, cm): " + " | ".join(f"t={t}: {np.mean([xyd(e, t) for e in eps]) * 100:.0f}+-{np.std([xyd(e, t) for e in eps]) * 100:.0f}" for t in (0, 10, 20, 30, 40, 50, 60)))
    t_close = np.array([e["tc"] for e in eps]); print(f"   close timestep: mean {t_close.mean():.1f} std {t_close.std():.1f} | quantiles 5/25/50/75/95: {np.quantile(t_close, [.05, .25, .5, .75, .95]).tolist()}")
    refs = np.array([e["ref"] for e in eps]); eta = lambda y: float(sum((refs == r).sum() * (y[refs == r].mean() - y.mean()) ** 2 for r in set(refs)) / ((y - y.mean()) ** 2).sum())
    z30 = np.array([e["s"][30, 2] for e in eps]); xy_anchor = np.array([e["anchor"][:2] for e in eps])
    print(f"   target-identity effect (eta^2 over {len(set(refs))} referents): z(t=30) {eta(z30):.3f} | z at close {eta(zc):.3f} | t_close {eta(t_close.astype(float)):.3f}   (chance level ~ {len(set(refs)) / len(refs):.3f})")
    print(f"   R^2 of z(t=30) on target XY (linear): {r2(z30, xy_anchor):.3f} | R^2 of t_close on target XY: {r2(t_close.astype(float), xy_anchor):.3f}")
    # frame-level regressions: future z
    rows = []
    for e in eps:
        T = len(e["a"])
        for t in range(0, min(e["tc"] + 15, T - 50), 2): rows.append(dict(t=t, tau=t - e["tc"], sz=e["s"][t, 2], gap=e["s"][t, 2] - e["anchor"][2], xyd=xyd(e, t), grip=e["s"][t, 6], tz=e["anchor"][2], y1=e["a"][t + 1, 2], y5=e["a"][t:t + 5, 2].mean(), y10=e["a"][t:t + 10, 2].mean(), yend=e["a"][t + 49, 2], ymin=e["a"][t:t + 50, 2].min()))
    R = pd.DataFrame(rows); print(f"\n== STEP 1 (demonstrations): R^2 of future action z on candidate predictors, {len(R)} frames of poker demos (t up to close+15) ==")
    for yk, lab in (("y1", "next action z"), ("y5", "first-5 mean z"), ("y10", "first-10 mean z"), ("yend", "chunk endpoint z (t+49)"), ("ymin", "min z in chunk")):
        y = R[yk].values; print(f"   {lab:26s} | state z (binned) {binned_r2(y, R.sz.values):.3f} | vertical gap (=state z - target z) {binned_r2(y, R.gap.values):.3f} | target z {binned_r2(y, R.tz.values + 1e-9 * np.random.default_rng(0).normal(size=len(R))):.3f} | timestep t {binned_r2(y, R.t.values):.3f} | XY dist to target {binned_r2(y, R.xyd.values):.3f} | gripper state {r2(y, R.grip.values):.3f} | state z + t + XYdist (OLS) {r2(y, R[['sz', 't', 'xyd']].values):.3f}")
    print("   (target z is constant in poker -> any apparent R^2 for it is numerical noise)")
    for task in ("select_book", "select_mahjong", "select_drink", "select_toy", "select_fruit"):
        g = E[E.task == task]; z_all = np.array([ (lambda ep: d.hf_dataset[int(fi[ep]) + int(r.t_close)]["observation.state"][2])(int(r.episode)) for _, r in g.iloc[:100].iterrows()]); res[task]["ee_z_at_close_mean_100eps"] = float(z_all.mean()); res[task]["ee_z_at_close_std_100eps"] = float(z_all.std())
    print("\n== across tasks: EE z at close vs target z (100 episodes each) -> is target z informative ACROSS tasks? ==")
    for task, v in res.items(): print(f"   {task:15s} target z {v['anchor_z_mean']:.3f} | EE z at close {v.get('ee_z_at_close_mean_100eps', float('nan')):.3f} +- {v.get('ee_z_at_close_std_100eps', float('nan')) * 1000:.1f}mm")
    json.dump({"tasks": res}, open(OUT / "g10_dataset_audit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
