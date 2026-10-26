#!/usr/bin/env python
"""G14 analysis. Runs: log (A, no intervention, must equal the G5 raw G4d rollouts), open (B, wrist robot pixels replaced by an OPEN-looking matched donor when physical gripper closed and EE-GT 3D > 6cm; state bit untouched), closed (C, control: closed-looking donor).
closed-and-far (CF) step = physical gripper state closed (state > 0.5) AND EE-to-GT-keypoint 3D distance > 6cm. dz_k = planned chunk z_k - EE z (cm). e_z magnitude = sqrt(d3^2 - dxy^2) (EE above the target in all CF frames of interest). rho_k = dz_k / (-e_z)."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g5_analysis import analyse  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g14"); G5D = Path("outputs/eval/safediff_vla_g5_closed_loop/g4d"); MODES = ("log", "open", "closed"); LAB = {"log": "A raw", "open": "B open-donor", "closed": "C closed-donor (control)"}; FAR = 0.06; THR = 0.5; SEEDS = list(range(3000, 3010)); BASE = np.array([0.0, -0.4, 0.78])


def ci(g): c = g <= THR; return int(np.argmax(c)) if c.any() else 50


def main():
    L = {m: pickle.load(open(OUT / m / "g4d" / "g14_log.pkl", "rb")) for m in MODES}; E = {m: json.load(open(OUT / m / "g4d" / "episodes_raw.json")) for m in MODES}; G5e = json.load(open(G5D / "episodes_raw.json"))
    print("== validation: A (logging only) vs the original G5 raw-G4d rollouts ==")
    d = [np.abs(np.array([s['ee_pos'] for s in a['steps']]) - np.array([s['ee_pos'] for s in b['steps']])).max() if len(a['steps']) == len(b['steps']) else 99 for a, b in zip(E["log"], G5e)]; print(f"   max |EE diff| over 10 episodes = {max(d) * 100:.3f} cm; success identical: {[a['success'] == b['success'] for a, b in zip(E['log'], G5e)].count(True)}/10")
    by = {m: {s: [r for r in L[m]["log"] if r["ep"] == i] for i, s in enumerate(SEEDS)} for m in MODES}
    ana = {m: {e["seed"]: analyse(e) for e in E[m]} for m in MODES}; tpos = {m: {e["seed"]: np.array([st["target_pos"] for st in e["steps"]])[:, :2] - BASE[:2] for e in E[m]} for m in MODES}
    cf = lambda r: r["grip"] > 0.5 and r["d3"] > FAR
    # ---------------- intervened chunks, paired ----------------
    print("\n== INTERVENED REPLANS: planned chunk with the patched wrist image vs the counterfactual chunk of the UNPATCHED image (same observation, same anchor pipeline) ==")
    for m in ("open", "closed"):
        iv = [r for s in SEEDS for r in by[m][s] if r["raw_chunk"] is not None]
        if not iv: print(f"   {LAB[m]}: no intervened replans"); continue
        ez = lambda r: -np.sqrt(max(r["d3"] ** 2 - r["dxy"] ** 2, 1e-9)); dz = lambda r, c, k: (c["xyz"][k, 2] - r["ee"][2]) * 100
        print(f"  {LAB[m]}: n={len(iv)} intervened replans in {len({r['ep'] for r in iv})} episodes")
        for k, lab in ((9, "dz10"), (19, "dz20"), (29, "dz30"), (49, "endpoint dz")): print(f"     {lab:12s} raw {np.mean([dz(r, r['raw_chunk'], k) for r in iv]):+6.2f} -> patched {np.mean([dz(r, r['chunk'], k) for r in iv]):+6.2f} cm")
        print("     rho_10/20/30/50  raw " + " ".join(f"{np.median([(r['raw_chunk']['xyz'][k, 2] - r['ee'][2]) / ez(r) for r in iv]):+.2f}" for k in (9, 19, 29, 49)) + " | patched " + " ".join(f"{np.median([(r['chunk']['xyz'][k, 2] - r['ee'][2]) / ez(r) for r in iv]):+.2f}" for k in (9, 19, 29, 49)))
        print(f"     predicted first close idx raw {np.mean([ci(r['raw_chunk']['grip']) for r in iv]):.1f} -> patched {np.mean([ci(r['chunk']['grip']) for r in iv]):.1f} | chunk closed fraction raw {np.mean([(r['raw_chunk']['grip'] <= THR).mean() for r in iv]):.2f} -> {np.mean([(r['chunk']['grip'] <= THR).mean() for r in iv]):.2f} | |dXY| of the chunk {np.mean([np.linalg.norm(r['chunk']['xyz'][:, :2] - r['raw_chunk']['xyz'][:, :2], axis=1).mean() for r in iv]) * 100:.2f} cm | fraction of intervened chunks that descend (dz30 <= -3cm): raw {np.mean([dz(r, r['raw_chunk'], 29) <= -3 for r in iv]):.2f} -> patched {np.mean([dz(r, r['chunk'], 29) <= -3 for r in iv]):.2f}")
    # ---------------- closed-and-far chunks across runs ----------------
    print("\n== ALL closed-and-far replans (physical gripper closed, EE-GT > 6cm): what the decoder plans and what the EE then does in the next 50 steps ==")
    def outcome(m, s, r):
        rows = by[m][s]; k = r["t"]; tgt = tpos[m][s]; e0 = r["ee"]; j = min(k + 50, len(rows) - 1); e1 = rows[j]["ee"]; mv = e1[:2] - e0[:2]; dg = tgt[min(k, len(tgt) - 1)] - e0[:2]
        return {"dd3": (r["d3"] - rows[j]["d3"]) * 100, "cos": float(mv @ dg / (np.linalg.norm(mv) * np.linalg.norm(dg) + 1e-9))}
    for m in MODES:
        rr = [(s, r) for s in SEEDS for r in by[m][s] if r["chunk"] is not None and cf(r)]; rr_iv = [(s, r) for s, r in rr if r["patched"]]; rr_no = [(s, r) for s, r in rr if not r["patched"]]
        for lab, xs in (("all CF replans", rr), ("  of which patched", rr_iv), ("  of which unpatched (no donor)", rr_no)):
            if not xs: continue
            dzz = lambda r, k: (r["chunk"]["xyz"][k, 2] - r["ee"][2]) * 100; oc = [outcome(m, s, r) for s, r in xs]
            print(f"   {LAB[m]:26s} {lab:30s} n={len(xs):3d} | dz10/20/30/end {np.mean([dzz(r, 9) for _, r in xs]):+5.1f}/{np.mean([dzz(r, 19) for _, r in xs]):+5.1f}/{np.mean([dzz(r, 29) for _, r in xs]):+5.1f}/{np.mean([dzz(r, 49) for _, r in xs]):+5.1f} cm | descend(dz30<=-3cm) {np.mean([dzz(r, 29) <= -3 for _, r in xs]):.2f} | first close idx {np.mean([ci(r['chunk']['grip']) for _, r in xs]):.1f} | closed frac {np.mean([(r['chunk']['grip'] <= THR).mean() for _, r in xs]):.2f} | next-50-step 3D distance reduction {np.mean([o['dd3'] for o in oc]):+5.1f}cm | XY cos toward target {np.mean([o['cos'] for o in oc]):+.2f}")
    # ---------------- episode-level feedback-loop metrics ----------------
    print("\n== FEEDBACK-LOOP METRICS per episode: closed-and-far steps | longest consecutive CF run | reopen events (physical closed->open) | first step with EE-GT <3cm | % of steps physical-closed while >6cm | CF replans with descent (dz30<=-3cm) | patched steps / trigger steps ==")
    summ = {m: [] for m in MODES}
    for s in SEEDS:
        line = f" seed {s}"
        for m in MODES:
            rows = by[m][s]; n = len(rows); c = np.array([cf(r) for r in rows]); g = np.array([r["grip"] > 0.5 for r in rows]); d3 = np.array([r["d3"] for r in rows]); runs = 0; best = 0
            for v in c: runs = runs + 1 if v else 0; best = max(best, runs)
            reopen = int(np.sum(g[:-1] & ~g[1:])); t3 = int(np.argmax(d3 < 0.03)) if (d3 < 0.03).any() else None; desc = sum(1 for r in rows if r["chunk"] is not None and cf(r) and (r["chunk"]["xyz"][29, 2] - r["ee"][2]) * 100 <= -3); trig = sum(1 for r in rows if r["trig"]); pat = sum(1 for r in rows if r["patched"])
            frac = float(np.mean((g & (d3 > FAR)))) * 100; summ[m].append({"cf": int(c.sum()), "run": best, "reopen": reopen, "t3": t3, "frac": frac, "desc": desc, "pat": pat, "trig": trig, "n": n, "mind3": d3.min() * 100})
            line += f" | {m[:4]}: CF {int(c.sum()):3d} run {best:3d} reop {reopen} t<3cm {str(t3):>4s} {frac:4.0f}% desc {desc} pat {pat}/{trig}"
        print(line)
    print("   MEAN over 10 seeds:")
    for m in MODES:
        x = summ[m]; t3 = [e["t3"] if e["t3"] is not None else 500 for e in x]; print(f"     {LAB[m]:26s} CF steps {np.mean([e['cf'] for e in x]):6.1f} | longest CF run {np.mean([e['run'] for e in x]):6.1f} | reopen events {np.mean([e['reopen'] for e in x]):.2f} | episodes reaching <3cm {sum(e['t3'] is not None for e in x)}/10 (mean first step, none=500: {np.mean(t3):.0f}) | physical-closed while >6cm {np.mean([e['frac'] for e in x]):.1f}% of steps | CF descent chunks {np.mean([e['desc'] for e in x]):.2f} | min EE-GT 3D {np.mean([e['mind3'] for e in x]):.1f}cm | patched/trigger {sum(e['pat'] for e in x)}/{sum(e['trig'] for e in x)}")
    # ---------------- secondary ----------------
    print("\n== SECONDARY (n=10; mechanism evidence, no performance claim) ==")
    rows = [("wrong card displaced", lambda a: a["wrong_card_displaced"]), ("reactive close occurred", lambda a: a["close"] is not None), ("close near the correct card (of closes)", None), ("target lifted >3cm", lambda a: a["target_max_lift_z"] > .03), ("native success", lambda a: a["success"]), ("min EE-GT xy (cm)", lambda a: a["min_ee_gt_xy"] * 100), ("min EE-GT 3D (cm)", lambda a: a["min_ee_gt_3d"] * 100)]
    print(f"   {'':40s}" + "".join(f"{LAB[m]:>26s}" for m in MODES))
    for lab, f in rows:
        vals = []
        for m in MODES:
            if f is None: cl = [a["close"]["correct"] for a in ana[m].values() if a["close"]]; vals.append(f"{sum(cl)}/{len(cl)}")
            else: v = np.mean([f(a) for a in ana[m].values()]); vals.append(f"{v:.2f}" if "cm" not in lab else f"{v:.2f}")
        print(f"   {lab:40s}" + "".join(f"{v:>26s}" for v in vals))
    print("   successes:", {m: [s for s, a in ana[m].items() if a["success"]] for m in MODES})
    pickle.dump({"summ": summ}, open(OUT / "g14_summary.pkl", "wb"))


if __name__ == "__main__":
    main()
