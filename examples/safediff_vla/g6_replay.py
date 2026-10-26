#!/usr/bin/env python
"""G6 STEP 1/2: offline replay of the stabiliser on the existing G5 G4d logs (NO control effect: the robot trajectory is the raw one; no success / future-observation claims).
Per replan we use only logged inference-side quantities: candidate boxes, fine probabilities p, selected crop, alpha-weighted fine location of the SELECTED candidate (the only candidate whose alpha location was logged), final map w2 (map_ok = final front argmax cell centre inside the selected crop).
GT card positions are used only to score identity. Identity of a location = nearest GT card in front-image pixels. Because only the selected candidate has a logged fine location, the replay's lock location L is updated from the selected candidate's fine location when it is the associated candidate, else kept (cards static) -- an approximation that the closed-loop run does not need.
Anchor xyz cannot be recomputed offline (needs the model); the offline anchor proxy = table-plane projection of the identity location (front ray at the card-surface plane), compared for RAW (selected fine location) and STABILISED (L) alike."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g4b_geometry import CAL, proj  # noqa: E402
from g4b_proposals import to_plane  # noqa: E402
from g6_stabilizer import Stabilizer  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g5_closed_loop")


def card_px(cands_world): return {k: np.array(proj(CAL[0], v)) for k, v in cands_world.items()}


def nearest_px(px, loc): k = min(px, key=lambda kk: np.linalg.norm(px[kk] - loc)); return k, float(np.linalg.norm(px[k] - loc))


def replay_episode(ep):
    S = ep["steps"]; tgt = ep["meta"]["target_entity"]; stab = Stabilizer(); rows = []
    for i in [i for i in range(1, len(S)) if S[i]["planner_replanned_this_step"]]:
        g = S[i]["grounder"]; d = i - 1; cw = S[d]["candidates_world"]; px = card_px(cw); sel = g["selected"]; fp = np.array(g["fine_px_front"]); c = g["final_front_argmax"]; mc = np.array([(c % 8 + .5) * 60, (c // 8 + .5) * 60])
        cands = []
        for j, (b, p) in enumerate(zip(g["boxes"], g["p"])):
            x0, y0, s = b; cands.append({"box": tuple(b), "p": float(p), "fine": tuple(fp) if j == sel else None, "map_ok": bool(j == sel and x0 <= mc[0] <= x0 + s and y0 <= mc[1] <= y0 + s)})
        dec = stab.step(cands); raw_loc = fp; st_loc = stab.L if stab.locked else fp
        gtw = np.array(cw[tgt]); rk, _ = nearest_px(px, raw_loc); sk, _ = nearest_px(px, st_loc)
        err = lambda loc: float(np.linalg.norm(to_plane(0, *loc)[:2] - gtw[:2]))
        anchor_card = min(cw, key=lambda k: np.linalg.norm(np.array(cw[k])[:2] - np.array(S[i]["planner_target_world"])[:2]))
        rows.append({"i": i, "action": dec["action"], "locked": stab.locked, "raw_card": rk, "raw_correct": rk == tgt, "stab_card": sk, "stab_correct": sk == tgt, "raw_proxy_err": err(raw_loc), "stab_proxy_err": err(st_loc), "raw_anchor_card_correct": anchor_card == tgt, "p_sel": float(g["p"][sel]), "map_ok": cands[sel]["map_ok"],
                     "target_displaced_by_now": max(float(np.linalg.norm(np.array(S[k]["candidates_world"][tgt])[:2] - np.array(S[0]["candidates_world"][tgt])[:2])) for k in range(0, i)) > 0.03})
    return {"seed": ep["seed"], "target": tgt, "rows": rows, "len": len(S) - 1, "success": ep["success"]}


def main():
    eps = json.load(open(OUT / "g4d" / "episodes_raw.json")); R = [replay_episode(e) for e in eps]; full = [r for r in R if r["seed"] in (3000, 3003, 3005, 3006, 3007)]
    def acc(rs, key, lo, hi): xs = [row[key] for r in rs for j, row in enumerate(r["rows"]) if lo <= j < hi]; return (float(np.mean(xs)), len(xs)) if xs else (None, 0)
    print("== correctness of the identity location by replan index (RAW selected fine location vs STABILISED lock location) ==")
    for name, rs in (("5 long-run seeds (3000,3003,3005,3006,3007)", full), ("all 10 seeds", R)):
        print(" ", name)
        for lab, lo, hi in (("1st", 0, 1), ("2nd-3rd", 1, 3), ("4th-6th", 3, 6), ("7th+", 6, 99), ("all", 0, 99)):
            r_, n = acc(rs, "raw_correct", lo, hi); s_, _ = acc(rs, "stab_correct", lo, hi); a_, _ = acc(rs, "raw_anchor_card_correct", lo, hi)
            print(f"     {lab:8s} n={n:3d} | raw fine-card {r_:.2f} | stabilised {s_:.2f} | (raw planner-anchor card {a_:.2f})")
    sw = lambda rows, k: sum(1 for a, b in zip(rows[:-1], rows[1:]) if a[k] != b[k]); n = len(R)
    print("\n== identity switches per episode (consecutive replans) ==\n   raw fine-card %.2f | stabilised %.2f | raw planner-anchor card (G5 value) 3.0" % (np.mean([sw([x for x in r["rows"]], "raw_card") for r in R]), np.mean([sw([x for x in r["rows"]], "stab_card") for r in R])))
    print("   stabiliser actions:", {a: sum(1 for r in R for x in r["rows"] if x["action"] == a) for a in ("raw", "lock", "hold", "switch", "grace", "unlock")})
    print("\n== anchor proxy error (table-plane projection of identity location vs GT xy, cm) mean/median, by replan index ==")
    for lab, lo, hi in (("1st", 0, 1), ("2nd-3rd", 1, 3), ("4th-6th", 3, 6), ("7th+", 6, 99)):
        rr = [row for r in full for j, row in enumerate(r["rows"]) if lo <= j < hi]
        if rr: print(f"   {lab:8s} raw {np.mean([x['raw_proxy_err'] for x in rr]) * 100:5.1f}/{np.median([x['raw_proxy_err'] for x in rr]) * 100:5.1f} | stabilised {np.mean([x['stab_proxy_err'] for x in rr]) * 100:5.1f}/{np.median([x['stab_proxy_err'] for x in rr]) * 100:5.1f}")
    print("\n== LOCK bookkeeping / failure modes per episode ==")
    fm = {"A_correct_lock_stays_correct": [], "B_wrong_lock_persists": [], "C_correct_lock_lost": [], "D_lost_after_genuine_card_motion": [], "B2_wrong_lock_recovered": [], "never_locked": []}
    for r in R:
        rows = r["rows"]; li = next((j for j, x in enumerate(rows) if x["action"] == "lock"), None)
        if li is None: fm["never_locked"].append(r["seed"]); print(f"   seed {r['seed']}: never locked | raw correct {np.mean([x['raw_correct'] for x in rows]):.2f}"); continue
        first_ok = rows[li]["stab_correct"]; after = rows[li:]; frac = np.mean([x["stab_correct"] for x in after]); late = np.mean([x["stab_correct"] for x in after[1:]]) if len(after) > 1 else frac; moved = any(x["target_displaced_by_now"] for x in after)
        if first_ok and late >= 0.8: cat = "A_correct_lock_stays_correct"
        elif first_ok: cat = "D_lost_after_genuine_card_motion" if moved else "C_correct_lock_lost"
        elif late >= 0.8 or (frac >= 0.5 and any(x["stab_correct"] for x in after[1:])): cat = "B2_wrong_lock_recovered"
        else: cat = "B_wrong_lock_persists"
        fm[cat].append(r["seed"]); print(f"   seed {r['seed']}: locked at replan {li + 1} ({'correct' if first_ok else 'WRONG'}) | stab-correct after lock {frac:.2f} | raw-correct (all) {np.mean([x['raw_correct'] for x in rows]):.2f} | actions {''.join(x['action'][0] for x in rows)} | -> {cat}")
    print("   counts:", {k: len(v) for k, v in fm.items()}, "| episodes locked:", n - len(fm["never_locked"]), "| initially correct locks:", len(fm["A_correct_lock_stays_correct"]) + len(fm["C_correct_lock_lost"]) + len(fm["D_lost_after_genuine_card_motion"]), "| initially wrong locks:", len(fm["B_wrong_lock_persists"]) + len(fm["B2_wrong_lock_recovered"]))
    json.dump({"episodes": R, "failure_modes": fm}, open(OUT / "g6_replay.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
