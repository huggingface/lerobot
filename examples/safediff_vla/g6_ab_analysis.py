#!/usr/bin/env python
"""G6 closed-loop A/B: raw G4d (G5 logs) vs G4d + temporal referent stabilisation (same seeds 3000-3009, decoder / reactive close / phase logic identical). Paired by seed; trajectories diverge after the first replan, so per-episode outcomes are descriptive only.
Identity of a location = nearest GT card in front-image pixels (evaluation only)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g4b_geometry import CAL, proj  # noqa: E402
from g5_analysis import analyse, OUT  # noqa: E402

LONG5 = (3000, 3003, 3005, 3006, 3007)


def px_card(cw, loc): px = {k: np.array(proj(CAL[0], v)) for k, v in cw.items()}; return min(px, key=lambda k: np.linalg.norm(px[k] - np.asarray(loc)))


def ident_rows(ep):
    S = ep["steps"]; tgt = ep["meta"]["target_entity"]; rows = []
    for i in [i for i in range(1, len(S)) if S[i]["planner_replanned_this_step"]]:
        g = S[i]["grounder"]; cw = S[i - 1]["candidates_world"]; st = g.get("stab"); loc = st["L"] if (st and st["locked"] and st["L"] is not None) else g["fine_px_front"]
        rows.append({"i": i, "id_card": px_card(cw, loc), "id_correct": px_card(cw, loc) == tgt, "action": st["action"] if st else "raw", "used": st["used"] if st else "raw", "locked": bool(st["locked"]) if st else False})
    return rows


def main():
    raw_eps = {e["seed"]: e for e in json.load(open(OUT / "g4d" / "episodes_raw.json"))}; st_eps = {e["seed"]: e for e in json.load(open(OUT / "g4d_stab" / "episodes_raw.json"))}; seeds = sorted(raw_eps); A = {"raw": {s: analyse(raw_eps[s]) for s in seeds}, "stab": {s: analyse(st_eps[s]) for s in seeds}}
    I = {"raw": {s: ident_rows(raw_eps[s]) for s in seeds}, "stab": {s: ident_rows(st_eps[s]) for s in seeds}}; M = ("raw", "stab")
    print("== PAIRED PER-SEED (raw G4d | G4d+stab): plan1 ok | replan-correct (anchor nearest-card) | identity-correct | planner switches | minEE-GT xy cm | close | wrong-card displaced | success | len ==")
    for s in seeds:
        line = f" seed {s} {raw_eps[s]['meta']['target_entity']:>17s}"
        for m in M:
            a = A[m][s]; ir = I[m][s]; c = a["close"]; line += f" | {m:4s} {'OK ' if a['first_plan_correct'] else 'BAD'} {a['n_correct_replans']:2d}/{a['n_replans']:2d} id {np.mean([r['id_correct'] for r in ir]):.2f} sw{a['planner_card_switches']} min{a['min_ee_gt_xy'] * 100:4.1f} close {('%s' % ('OK' if c['correct'] else 'wr')) if c else '--'} wd{int(a['wrong_card_displaced'])} {'SUCC' if a['success'] else 'fail'} {a['len']}"
        print(line)
    f = lambda m, key, ss=seeds: np.mean([key(A[m][s]) for s in ss])
    print("\n== SUMMARY (10 paired seeds) ==")
    rows = [("first-plan target card correct", lambda e: e["first_plan_correct"]), ("first-plan dist xy (cm)", lambda e: e["first_plan_dist_xy"] * 100), ("all-replan target-correct rate (anchor card)", lambda e: e["n_correct_replans"] / e["n_replans"]), ("planner card switches / episode", lambda e: e["planner_card_switches"]),
            ("min EE-GT xy (cm) mean", lambda e: e["min_ee_gt_xy"] * 100), ("min EE-GT 3D (cm) mean", lambda e: e["min_ee_gt_3d"] * 100), ("closest-approached card is target", lambda e: e["closest_card_is_target"]), ("target card displaced", lambda e: e["target_displaced"]), ("wrong card displaced", lambda e: e["wrong_card_displaced"]),
            ("close occurred", lambda e: e["close"] is not None), ("target max lift >3cm", lambda e: e["target_max_lift_z"] > .03), ("native is_grasped(target) ever", lambda e: e["target_is_grasped_any(native)"]), ("native success (secondary; n=10)", lambda e: e["success"])]
    print(f"   {'':46s}{'raw':>9s}{'stab':>9s}")
    for name, k in rows: print(f"   {name:46s}" + "".join(f"{f(m, k):9.3f}" for m in M))
    for m in M: cl = [A[m][s]["close"]["correct"] for s in seeds if A[m][s]["close"]]; print(f"   close near correct card ({m}): {sum(cl)}/{len(cl)}")
    print("\n== replan correctness by replan index. planner-anchor nearest-card / identity location (fine or lock) ==")
    for name, ss in (("all 10 seeds", seeds), ("5 long-run seeds of G5 (3000,3003,3005,3006,3007)", LONG5)):
        print("  ", name)
        for lab, lo, hi in (("1st", 0, 1), ("2nd-3rd", 1, 3), ("4th-6th", 3, 6), ("7th+", 6, 99)):
            out = []
            for m in M:
                xa = [r["target_correct"] for s in ss for j, r in enumerate(A[m][s]["replans"]) if lo <= j < hi]; xi = [r["id_correct"] for s in ss for j, r in enumerate(I[m][s]) if lo <= j < hi]; out.append(f"{m}: anchor {np.mean(xa):.2f} id {np.mean(xi):.2f} (n={len(xa)})" if xa else f"{m}: -")
            print(f"     {lab:8s} " + " | ".join(out))
    print("\n== STABILITY (monitor-only per-frame target, stab uses the current lock read-only) ==")
    for m in M:
        ftd = np.concatenate([np.diff(np.array([x["monitor_target_world"] for x in (raw_eps if m == 'raw' else st_eps)[s]["steps"][1:]])[:, :2], axis=0).__pow__(2).sum(1) ** .5 for s in seeds]) * 100
        sw = np.mean([sum(1 for a_, b_ in zip(ids[:-1], ids[1:]) if a_ != b_) for ids in [[min((raw_eps if m == 'raw' else st_eps)[s]["steps"][i]["candidates_world"], key=lambda k: np.linalg.norm(np.array((raw_eps if m == 'raw' else st_eps)[s]["steps"][i]["candidates_world"][k])[:2] - np.array((raw_eps if m == 'raw' else st_eps)[s]["steps"][i + 1]["monitor_target_world"])[:2])) for i in range(len((raw_eps if m == 'raw' else st_eps)[s]["steps"]) - 1)] for s in seeds]])
        print(f"   {m:4s} per-frame monitor-target displacement median/mean/p90 {np.median(ftd):.2f}/{ftd.mean():.2f}/{np.percentile(ftd, 90):.2f} cm | monitor card switches/episode {sw:.1f}")
    print("\n== LOCK bookkeeping (stab run) ==")
    st_rows = I["stab"]; locked_eps = [s for s in seeds if any(r["action"] == "lock" for r in st_rows[s])]; init_ok, init_bad, recov, cats = [], [], [], {}
    for s in seeds:
        rows_ = st_rows[s]; li = next((j for j, r in enumerate(rows_) if r["action"] == "lock"), None)
        if li is None: cats[s] = "never_locked"; continue
        ok0 = rows_[li]["id_correct"]; after = rows_[li:]; late = np.mean([r["id_correct"] for r in after[1:]]) if len(after) > 1 else float(ok0); moved = any(x["target_displaced_by_now"] if "target_displaced_by_now" in x else False for x in after)
        (init_ok if ok0 else init_bad).append(s); sw_ = [r for r in after if r["action"] == "switch"]
        if ok0: cats[s] = "A_correct_lock_stays_correct" if late >= .8 else "C_correct_lock_lost"
        else: cats[s] = "B2_wrong_lock_recovered" if (late >= .8 or any(r["id_correct"] for r in after[1:])) else "B_wrong_lock_persists"; recov += [s] if cats[s] == "B2_wrong_lock_recovered" else []
        print(f"   seed {s}: lock@replan {li + 1} ({'correct' if ok0 else 'WRONG'}) | id-correct after lock {np.mean([r['id_correct'] for r in after]):.2f} | actions {''.join(r['action'][0] for r in rows_)} | switches {len(sw_)} | -> {cats[s]}")
    cnt = {}; [cnt.__setitem__(v, cnt.get(v, 0) + 1) for v in cats.values()]; print("   LOCKED episodes:", len(locked_eps), "| initially correct locks:", len(init_ok), "| initially wrong locks:", len(init_bad), "(seeds", init_bad, ") | wrong-lock recoveries:", len(recov), "| categories:", cnt)
    print("   action totals:", {a: sum(1 for s in seeds for r in st_rows[s] if r["action"] == a) for a in ("lock", "hold", "switch", "grace", "unlock", "raw")}, "| forced-candidate uses:", sum(1 for s in seeds for r in st_rows[s] if r["used"] == "forced_candidate"), "| grace anchor reuse:", sum(1 for s in seeds for r in st_rows[s] if r["used"] == "grace_prev_anchor"))
    print("\n== ANCHOR COUPLING AFTER STABILISATION: replans with CORRECT identity -> anchor error ==")
    for m in M:
        rr = [(r, a_) for s in seeds for r, a_ in zip(I[m][s], A[m][s]["replans"]) if r["id_correct"]]; e = np.array([a_["planner_to_gt_xy"] for _, a_ in rr]) * 100
        print(f"   {m:4s} identity-correct replans n={len(rr)}: anchor err median {np.median(e):.1f} mean {e.mean():.1f} cm | >5cm {np.mean(e > 5):.2f} | >10cm {np.mean(e > 10):.2f}")
    for m in M:
        cs = [r for s in seeds for r in A[m][s]["replans"] if "case" in r]
        print(f"   {m:4s} map-based cases: " + ", ".join(f"case{c}: {sum(1 for r in cs if r['case'] == c)}/{len(cs)}" for c in (1, 2, 3)))
    json.dump({"A": {m: {str(s): {k: v for k, v in A[m][s].items() if not k.startswith('_')} for s in seeds} for m in M}, "ident": {m: {str(s): I[m][s] for s in seeds} for m in M}}, open(OUT / "g6_ab_analysis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
