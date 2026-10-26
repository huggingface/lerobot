#!/usr/bin/env python
"""G12 analysis: gripper-state causal intervention on fixed observations (797 frames; primary subset = target >= 6cm below the EE, GT anchor A1). dz_k = chunk z_k - EE z (cm); rho_k = dz_k / e_z with e_z = anchor z - EE z.
Effects are paired per frame: OPEN-conditioned minus CLOSED-conditioned chunk of the SAME observation. 95% CIs: bootstrap over episodes (resampling (run, episode) clusters). Gripper action convention: 0 = closed, 1 = open (closed if <= 0.5)."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
O12 = Path("outputs/eval/safediff_vla_g12_audit"); THR = 0.5; rng = np.random.default_rng(0); ORDER = ("EARLY", "MID", "LATE", "NEAR/CONTACT")


def stratum(f):
    if f["ee_gt_xy"] < 0.08 or f["any_card_displaced_cm"] > 1.5: return "NEAR/CONTACT"
    if f["t"] <= 20: return "EARLY"
    return "LATE" if f["ee_gt_xy"] <= 0.12 else "MID"


def dz(f, c, sl): return (f["cond"][c]["xyz"][sl, 2] - f["ee_robot"][2]).mean() * 100
def rho(f, c, k): return (f["cond"][c]["xyz"][k, 2] - f["ee_robot"][2]) / (f["gt_anchor_robot"][2] - f["ee_robot"][2])
def closed(f, c): return f["cond"][c]["grip"] <= THR


def boot(vals, clusters, n=1000):
    vals = np.asarray(vals, float); u = np.unique(clusters); idx = {k: np.where(clusters == k)[0] for k in u}; m = []
    for _ in range(n): pick = rng.choice(u, len(u)); m.append(np.concatenate([vals[idx[k]] for k in pick]).mean())
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    F = []
    for src in ("g4d", "oracle"):
        for fr in pickle.load(open(O12 / src / "audit12.pkl", "rb")): fr["src"] = src; fr["stratum"] = stratum(fr); fr["cl"] = f"{src}{fr['ep']}"; F.append(fr)
    P = [f for f in F if f["ee_gt_z"] > 0.06]; cl = np.array([f["cl"] for f in P]); orig_closed = np.array([f["grip_raw"] > 0.5 for f in P])
    print(f"frames {len(F)} | primary subset (target >=6cm below EE) n={len(P)} | mean e_z {-np.mean([f['ee_gt_z'] for f in P]) * 100:.1f}cm | originally CLOSED {orig_closed.sum()} / OPEN {(~orig_closed).sum()} | raw gripper state values (rounded): {sorted(set(np.round([f['grip_raw'] for f in F], 1)))}")
    hz = [("first action", slice(0, 1)), ("first5", slice(0, 5)), ("first10", slice(0, 10)), ("step20", slice(19, 20)), ("step30", slice(29, 30)), ("endpoint", slice(49, 50))]; conds = ("C0", "A_open", "A_closed", "B_open", "B_closed")
    print("\n== PRIMARY (all primary-subset frames): vertical chunk, mean dz = chunk z - EE z (cm) and median closing ratio rho_k ==")
    print(f"   {'':10s}" + "".join(f"{h:>14s}" for h, _ in hz) + f"{'min dz':>10s} | rho_10/20/30/50")
    for c in conds: print(f"   {c:10s}" + "".join(f"{np.mean([dz(f, c, s) for f in P]):14.2f}" for _, s in hz) + f"{np.mean([(f['cond'][c]['xyz'][:, 2].min() - f['ee_robot'][2]) * 100 for f in P]):10.2f} | " + " ".join(f"{np.median([rho(f, c, k) for f in P]):+.2f}" for k in (9, 19, 29, 49)))
    print("\n== PAIRED effect OPEN - CLOSED on the SAME observation (cm; negative = OPEN descends more), mean [95% CI over episodes] ==")
    for var in ("A", "B"):
        print(f"  {var} ({'decoder state vector only' if var == 'A' else 'decoder state + VLM state token, latents/feature recomputed'}):")
        for h, s in hz: m, lo, hi = boot([dz(f, f"{var}_open", s) - dz(f, f"{var}_closed", s) for f in P], cl); print(f"     {h:13s} {m:+7.2f} [{lo:+.2f}, {hi:+.2f}]")
        for k in (9, 19, 29, 49): m, lo, hi = boot([rho(f, f"{var}_open", k) - rho(f, f"{var}_closed", k) for f in P], cl); print(f"     rho_{k + 1:<2d}       {m:+7.2f} [{lo:+.2f}, {hi:+.2f}]   (OPEN median {np.median([rho(f, f'{var}_open', k) for f in P]):+.2f}, CLOSED median {np.median([rho(f, f'{var}_closed', k) for f in P]):+.2f})")
    print("\n== 2x2: ORIGINAL gripper state x FORCED state (B = consistent intervention, A in brackets): mean dz at step30 (cm) | median rho_30 ==")
    for lab, m in (("originally OPEN", ~orig_closed), ("originally CLOSED", orig_closed)):
        fs = [f for f, k in zip(P, m) if k]
        print(f"   {lab:18s} n={len(fs):3d} | original C0: dz30 {np.mean([dz(f, 'C0', slice(29, 30)) for f in fs]):+.1f} rho30 {np.median([rho(f, 'C0', 29) for f in fs]):+.2f} || forced OPEN: B dz30 {np.mean([dz(f, 'B_open', slice(29, 30)) for f in fs]):+.1f} rho30 {np.median([rho(f, 'B_open', 29) for f in fs]):+.2f} [A {np.mean([dz(f, 'A_open', slice(29, 30)) for f in fs]):+.1f} / {np.median([rho(f, 'A_open', 29) for f in fs]):+.2f}] || forced CLOSED: B dz30 {np.mean([dz(f, 'B_closed', slice(29, 30)) for f in fs]):+.1f} rho30 {np.median([rho(f, 'B_closed', 29) for f in fs]):+.2f} [A {np.mean([dz(f, 'A_closed', slice(29, 30)) for f in fs]):+.1f} / {np.median([rho(f, 'A_closed', 29) for f in fs]):+.2f}]")
    print("\n== STRATIFIED paired effect OPEN - CLOSED: mean dz30 effect (cm) and OPEN/CLOSED median rho_30 ==")
    bins = [("vertical gap 6-10cm", lambda f: 0.06 < f["ee_gt_z"] <= 0.10), ("10-20cm", lambda f: 0.10 < f["ee_gt_z"] <= 0.20), (">20cm", lambda f: f["ee_gt_z"] > 0.20)] + [(s, (lambda s_: lambda f: f["stratum"] == s_)(s)) for s in ORDER]
    for lab, sel in bins:
        fs = [f for f in P if sel(f)]
        if len(fs) < 5: print(f"   {lab:22s} n={len(fs)} (too few)"); continue
        r = []
        for var in ("A", "B"): r.append((np.mean([dz(f, f"{var}_open", slice(29, 30)) - dz(f, f"{var}_closed", slice(29, 30)) for f in fs]), np.median([rho(f, f"{var}_open", 29) for f in fs]), np.median([rho(f, f"{var}_closed", 29) for f in fs])))
        print(f"   {lab:22s} n={len(fs):3d} | A: effect {r[0][0]:+.2f}cm rho30 {r[0][1]:+.2f}/{r[0][2]:+.2f} | B: effect {r[1][0]:+.2f}cm rho30 {r[1][1]:+.2f}/{r[1][2]:+.2f}")
    print("\n== GRIPPER OUTPUT (action chunk) ==  first close idx (50 = none) | fraction of chunk closed | frac chunks fully closed | frac chunks with a re-open | mean reopen events")
    gi = lambda f, c: int(np.argmax(closed(f, c))) if closed(f, c).any() else 50
    for lab, fs in (("ALL frames", F), ("primary subset", P)):
        print(f"  {lab}:")
        for c in conds: print(f"     {c:10s} {np.mean([gi(f, c) for f in fs]):6.1f} | {np.mean([closed(f, c).mean() for f in fs]):.2f} | {np.mean([closed(f, c).all() for f in fs]):.2f} | {np.mean([(closed(f, c)[:-1] & ~closed(f, c)[1:]).any() for f in fs]):.2f} | {np.mean([(closed(f, c)[:-1] & ~closed(f, c)[1:]).sum() for f in fs]):.2f}")
    for var in ("A", "B"): print(f"   {var}: gripper chunk L1 |OPEN - CLOSED| mean {np.mean([np.abs(f['cond'][f'{var}_open']['grip'] - f['cond'][f'{var}_closed']['grip']).mean() for f in F]):.3f} (range 0-1) ; frames where the first close idx differs by >=5 steps: {np.mean([abs(gi(f, f'{var}_open') - gi(f, f'{var}_closed')) >= 5 for f in F]):.2f}")
    print("   self-reinforcement: P(chunk fully closed | input CLOSED) = B %.2f, A %.2f ; P(chunk fully closed | input OPEN) = B %.2f, A %.2f" % tuple(np.mean([closed(f, c).all() for f in F]) for c in ("B_closed", "A_closed", "B_open", "A_open")))
    for lab, m in (("originally OPEN frames, forced CLOSED", False), ("originally CLOSED frames, forced OPEN", True)):
        fs = [f for f in F if (f["grip_raw"] > 0.5) == m]; c = "B_open" if m else "B_closed"; print(f"   {lab}: n={len(fs)} | mean first close idx C0 {np.mean([gi(f, 'C0') for f in fs]):.1f} -> {c} {np.mean([gi(f, c) for f in fs]):.1f} | fraction closed C0 {np.mean([closed(f, 'C0').mean() for f in fs]):.2f} -> {np.mean([closed(f, c).mean() for f in fs]):.2f}")
    print("\n== XY CONTROL GUARD (all frames; OPEN vs CLOSED, same observation) ==")
    for var in ("A", "B"):
        o, c = f"{var}_open", f"{var}_closed"; pl = lambda f, k: np.linalg.norm(np.diff(f["cond"][k]["xyz"], axis=0), axis=1).sum() * 100
        def cosv(f, k): ee = f["ee_robot"][:2]; v = f["cond"][k]["xyz"][:10, :2].mean(0) - ee; d = f["gt_anchor_robot"][:2] - ee; return float(v @ d / (np.linalg.norm(v) * np.linalg.norm(d) + 1e-9))
        print(f"   {var}: endpoint |XY open - XY closed| {np.mean([np.linalg.norm(f['cond'][o]['xyz'][49, :2] - f['cond'][c]['xyz'][49, :2]) for f in F]) * 100:.2f}cm vs endpoint |Z open - Z closed| {np.mean([abs(f['cond'][o]['xyz'][49, 2] - f['cond'][c]['xyz'][49, 2]) for f in F]) * 100:.2f}cm ; full-chunk mean |XY diff| {np.mean([np.linalg.norm(f['cond'][o]['xyz'][:, :2] - f['cond'][c]['xyz'][:, :2], axis=1).mean() for f in F]) * 100:.2f}cm vs |Z diff| {np.mean([np.abs(f['cond'][o]['xyz'][:, 2] - f['cond'][c]['xyz'][:, 2]).mean() for f in F]) * 100:.2f}cm ; path length open {np.mean([pl(f, o) for f in F]):.1f} closed {np.mean([pl(f, c) for f in F]):.1f}cm ; cos(first10, dir to anchor) open {np.mean([cosv(f, o) for f in F]):+.2f} closed {np.mean([cosv(f, c) for f in F]):+.2f}")


if __name__ == "__main__":
    main()
