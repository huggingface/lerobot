#!/usr/bin/env python
"""G11-A/B: dataset identifiability audit of TARGET-RELATIVE conditioning on the 6-task training set (lerobot/vlabench_unified, 500 episodes/task; no network, no training of anything beyond linear / tree probes).
Per frame t (every 2nd step up to t_close+15): s = EE xyz (state), T = target anchor xyz (episodes.json; first open->close commanded xyz), e = T - s, g = gripper state, t = frame index.
Outputs: absolute action position and RESIDUAL r = action - s at horizons next / first5 / first10 / t+20 / t+30 / t+49 (indices clamped at the episode end, as the chunk padding does), plus time-to-close.
Predictor sets: A s | B T | C e | D s+T | E s+e | F s+e+g+t | A2 s+g+t (no target).  Linear: D == E exactly (e = T - s).  Tree probe (HistGradientBoosting): D != E because differences are not free for trees.
Split: by episode (80/20) within a task; LOTO = fit on the other 5 tasks, test on the held-out task."""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g11_audit"); OUT.mkdir(parents=True, exist_ok=True); TASKS = ["select_poker", "select_mahjong", "select_fruit", "select_drink", "select_toy", "select_book"]; HZ = {"next": (1, 1), "first5": (0, 5), "first10": (0, 10), "t+20": (20, 21), "t+30": (30, 31), "t+49": (49, 50)}
PRED = {"A s": lambda d: d["s"], "B T": lambda d: d["T"], "C e": lambda d: d["e"], "D s+T": lambda d: np.c_[d["s"], d["T"]], "E s+e": lambda d: np.c_[d["s"], d["e"]], "F s+e+g+t": lambda d: np.c_[d["s"], d["e"], d["g"], d["t"]], "A2 s+g+t": lambda d: np.c_[d["s"], d["g"], d["t"]]}


def build():
    f = OUT / "frames.npz"
    if f.exists(): z = np.load(f, allow_pickle=True); return {k: z[k] for k in z.files}
    d = LeRobotDataset("lerobot/vlabench_unified"); fi = d.meta.episodes["dataset_from_index"]; ti = d.meta.episodes["dataset_to_index"]; E = pd.read_json("outputs/eval/safediff_vla_phase_d0_training_data_audit/episodes.json"); R = {k: [] for k in ("task", "ep", "t", "tc", "s", "T", "g", "ttc") + tuple(f"{p}_{h}" for h in HZ for p in ("abs",))}
    for ti_, task in enumerate(TASKS):
        for _, r in E[E.task == task].iterrows():
            ep = int(r.episode); sl = d.hf_dataset[int(fi[ep]):int(ti[ep])]; S = np.array(sl["observation.state"]); A = np.array(sl["action"]); n = len(A); tc = int(r.t_close); T = np.array(r.anchor, float)
            for t in range(0, min(tc + 15, n - 1), 2):
                R["task"].append(ti_); R["ep"].append(ep); R["t"].append(t); R["tc"].append(tc); R["s"].append(S[t, :3]); R["T"].append(T); R["g"].append(S[t, 6]); R["ttc"].append(tc - t)
                for h, (a, b) in HZ.items(): idx = np.clip(np.arange(t + a, t + b), 0, n - 1); R[f"abs_{h}"].append(A[idx, :3].mean(0))
        print("loaded", task, flush=True)
    out = {k: np.array(v) for k, v in R.items()}; np.savez_compressed(f, **out); return out


def r2(y, p): return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def fit_eval(Xtr, Ytr, Xte, Yte, model):
    cols = []
    if model == "hgb" and len(Xtr) > 30000: sel = np.random.default_rng(1).choice(len(Xtr), 30000, replace=False); Xtr, Ytr = Xtr[sel], Ytr[sel]  # tree probe: 30k-row subsample (speed)
    for c in range(Ytr.shape[1]):
        m = LinearRegression() if model == "lin" else HistGradientBoostingRegressor(max_iter=60, max_depth=4, learning_rate=0.15); m.fit(Xtr, Ytr[:, c]); cols.append(m.predict(Xte))
    return np.stack(cols, 1)


def main():
    D = build(); N = len(D["t"]); D["e"] = D["T"] - D["s"]; ytab = {}
    for h in HZ: ytab[("abs", h)] = D[f"abs_{h}"]; ytab[("res", h)] = D[f"abs_{h}"] - D["s"]
    rng = np.random.default_rng(0); eps = np.unique(D["ep"]); test_eps = set(rng.choice(eps, int(0.2 * len(eps)), replace=False)); te = np.array([e in test_eps for e in D["ep"]]); tr = ~te; task = D["task"]; res = {}
    print(f"frames {N} | per task: " + ", ".join(f"{TASKS[i][7:]} {int((task == i).sum())}" for i in range(6)))
    def per_task_table(model, hs, parts):
        for comp, sl, lab in ((2, slice(2, 3), "z"), (None, slice(0, 2), "xy")):
            print(f"\n  [{model}] {lab}: R^2 within task (episode split 80/20), outputs {parts}")
            print(f"     {'task':10s} {'out':10s}" + "".join(f"{k:>11s}" for k in PRED))
            for ti_, tk in enumerate(TASKS):
                m_tr = tr & (task == ti_); m_te = te & (task == ti_)
                for part in parts:
                    for h in hs:
                        Y = ytab[(part, h)][:, sl]; row = []
                        for pk, pf in PRED.items(): row.append(r2(Y[m_te], fit_eval(pf(D)[m_tr], Y[m_tr], pf(D)[m_te], Y[m_te], model))); res[(model, lab, tk, part, h, pk)] = row[-1]
                        print(f"     {tk[7:]:10s} {part + ':' + h:10s}" + "".join(f"{v:11.3f}" for v in row))
    print("\n== G11-A/B (1) WITHIN-TASK, LINEAR (D == E by construction) ==")
    per_task_table("lin", ("first10", "t+20", "t+49"), ("abs", "res"))
    print("\n== (2) WITHIN-TASK, TREE PROBE (non-linear; D vs E can differ) ==")
    per_task_table("hgb", ("t+20", "t+49"), ("res",))
    # ---------------- pooled and LOTO ----------------
    print("\n== (3) LEAVE-ONE-TASK-OUT: fit on the other 5 tasks (all frames), test on the held-out task. R^2 of RESIDUAL r = action - EE ==")
    for model in ("lin", "hgb"):
        for comp, sl, lab in ((2, slice(2, 3), "z"), (None, slice(0, 2), "xy")):
            for h in ("t+20", "t+49"):
                print(f"\n  [{model}] {lab} residual at {h}:  held-out task | " + " ".join(f"{k:>10s}" for k in ("A s", "A2 s+g+t", "B T", "C e", "E s+e", "F s+e+g+t")))
                for ti_, tk in enumerate(TASKS):
                    Y = ytab[("res", h)][:, sl]; row = []
                    for pk in ("A s", "A2 s+g+t", "B T", "C e", "E s+e", "F s+e+g+t"): X = PRED[pk](D); p = fit_eval(X[task != ti_], Y[task != ti_], X[task == ti_], Y[task == ti_], model); row.append(r2(Y[task == ti_], p)); res[(model, "LOTO", lab, tk, h, pk)] = row[-1]
                    print(f"     {tk[7:]:14s} | " + " ".join(f"{v:10.3f}" for v in row))
    # ---------------- relation r_z[k] ~ e_z, closing ratio ----------------
    print("\n== (4) RESIDUAL-vs-ERROR RELATION: slope of r_z[k] on e_z (through origin) and the closing ratio rho_k = r_z[k] / e_z (frames with |e_z| > 3cm), by task and phase stratum ==")
    t, tc = D["t"], D["tc"]; strat = np.where(t < 20, 0, np.where(t < tc - 15, 1, np.where(t <= tc, 2, 3))); names = ["EARLY", "MID", "NEAR", "POST"]
    for ti_, tk in enumerate(TASKS):
        line = []
        for si in range(4):
            m = (task == ti_) & (strat == si) & (np.abs(D["e"][:, 2]) > 0.03)
            if m.sum() < 20: line.append(f"{names[si]}: n<20"); continue
            e = D["e"][m, 2]; parts = []
            for h in ("first10", "t+20", "t+30", "t+49"): r = ytab[("res", h)][m, 2]; parts.append(f"{h} slope {float(e @ r / (e @ e)):+.2f}/rho {np.median(r / e):+.2f}")
            line.append(f"{names[si]} n={int(m.sum())}: " + "; ".join(parts))
        print(f"   {tk[7:]:8s} " + "\n            ".join(line))
    print("   (e_z < 0 means the target is BELOW the EE; rho ~ fraction of the remaining vertical gap closed by the action k steps ahead; a task-independent rho means e_z is a common control variable)")
    print("\n== (5) Is absolute target z a task-ID proxy? fraction of target-z variance explained by task identity (all frames): %.3f ; of relative e_z variance: %.3f ; of s_z: %.3f ==" % tuple(1 - np.mean([np.var(v[task == i]) * (task == i).mean() for i in range(6)]) / np.var(v) if False else sum((task == i).mean() * (v[task == i].mean() - v.mean()) ** 2 for i in range(6)) / v.var() for v in (D["T"][:, 2], D["e"][:, 2], D["s"][:, 2])))
    json.dump({"|".join(map(str, k)): v for k, v in res.items()}, open(OUT / "g11_audit.json", "w"), indent=0)


if __name__ == "__main__":
    main()
