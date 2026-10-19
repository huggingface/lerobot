#!/usr/bin/env python
"""G5 analysis: V2 / G3 M1 / G4d closed-loop grounding transfer (select_poker, fixed seeds). GT (card world positions, target) is used for diagnostics only.
Conventions: record i>=1 carries the policy log of the select_action call that PRECEDED it; a replan at record i was decided from the observation of record i-1 (EE / card positions at i-1) and its chunk acts through records i..i+49.
XY distances (table plane) are the primary geometry (cards lie on the table); `ee_to_target_dist` (3D) is also reported.
Target identity of a predicted anchor = nearest same-type card (XY) at decision time. Map-selected card (G4d only) = card whose projected front/right cell holds the most final-map mass."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g4b_geometry import CAL, proj  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g5_closed_loop"); MODELS = ("v2", "g3", "g4d"); CHUNK = 50


def xy(a): return np.asarray(a, float)[:2]


def nearest(cands, p):
    d = {k: float(np.linalg.norm(xy(v) - xy(p))) for k, v in cands.items()}; k = min(d, key=d.get); return k, d[k]


def cell_of(u, v): return int(np.clip(v // 60, 0, 7)) * 8 + int(np.clip(u // 60, 0, 7))


def map_card(cands, w2):
    m = {}
    for k, pos in cands.items():
        fu, fv = proj(CAL[0], pos); ru, rv = proj(CAL[1], pos); m[k] = w2[cell_of(fu, fv)] + w2[64 + cell_of(ru, rv)]
    return max(m, key=m.get)


def analyse(ep):
    S = ep["steps"]; tgt = ep["meta"]["target_entity"]; c0 = S[0]["candidates_world"]; gt0 = c0[tgt]; ee = np.array([s["ee_pos"] for s in S]); n = len(S)
    rep = [i for i in range(1, n) if S[i]["planner_replanned_this_step"]]; out = {"seed": ep["seed"], "target": tgt, "n_cards": len(c0), "rank9": tgt.startswith("9_"), "success": bool(ep["success"]), "len": n - 1}
    out["in_training_strings"] = ep["meta"]["in_training_strings"]; out["lang"] = ep["language"].get("decoded")
    R = []
    for i in rep:
        d = i - 1; cands = S[d]["candidates_world"]; pt = S[i]["planner_target_world"]; k, dist = nearest(cands, pt); gt = cands[tgt]; e0 = ee[d]; e1 = ee[min(i + CHUNK - 1, n - 1)]; mv = xy(e1) - xy(e0); dg = xy(gt) - xy(e0); dp = xy(pt) - xy(e0)
        cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        r = {"i": i, "nearest_card": k, "target_correct": k == tgt, "planner_to_gt_xy": float(np.linalg.norm(xy(pt) - xy(gt))), "ee_gt_start_xy": float(np.linalg.norm(dg)), "ee_gt_end_xy": float(np.linalg.norm(xy(e1) - xy(gt))), "disp_xy": float(np.linalg.norm(mv)),
             "cos_disp_to_gt": cos(mv, dg), "cos_disp_to_planner": cos(mv, dp), "approach_gain_xy": float(np.linalg.norm(dg) - np.linalg.norm(xy(e1) - xy(gt)))}
        g = S[i]["grounder"]
        if g and "w2" in g:
            w2 = np.array(g["w2"]); r["map_card"] = map_card(cands, w2); r["map_correct"] = r["map_card"] == tgt; r["anchor_le5"] = r["planner_to_gt_xy"] < 0.05; r["case"] = 1 if (r["map_correct"] and r["anchor_le5"]) else (2 if r["map_correct"] else 3)
            bx = g["boxes"]; fp = np.array(g["fine_px_front"]); r["fine_selected_src"] = g["src"][g["selected"]]
            # which card is closest (front pixel) to the alpha-weighted fine location
            pix = {kk: np.array(proj(CAL[0], v)) for kk, v in cands.items()}; r["fine_card"] = min(pix, key=lambda kk: np.linalg.norm(pix[kk] - fp)); r["fine_correct"] = r["fine_card"] == tgt
        R.append(r)
    out["replans"] = R
    # A: first plan
    r0 = R[0]; out["first_plan_correct"] = r0["target_correct"]; out["first_plan_dist_xy"] = r0["planner_to_gt_xy"]; out["first_chunk"] = {k: r0[k] for k in ("cos_disp_to_gt", "approach_gain_xy", "disp_xy", "ee_gt_start_xy", "ee_gt_end_xy")}
    out["first_chunk_toward_target"] = bool(r0["approach_gain_xy"] > 0.02 and r0["cos_disp_to_gt"] > 0.5)
    # C: min distances
    dxy = np.linalg.norm(ee[:, :2] - xy(gt0)[None], axis=1); d3 = np.array([s["ee_to_target_dist"] for s in S]); out["min_ee_gt_xy"] = float(dxy.min()); out["min_ee_gt_3d"] = float(d3.min()); out["t_min_xy"] = int(dxy.argmin())
    mins = {k: float(np.linalg.norm(ee[:, :2] - xy(v)[None], axis=1).min()) for k, v in c0.items()}; out["closest_approached_card"] = min(mins, key=mins.get); out["closest_card_is_target"] = out["closest_approached_card"] == tgt; out["min_ee_xy_per_card"] = mins
    # interaction: displaced cards
    disp = {}
    for k, v in c0.items(): disp[k] = max(float(np.linalg.norm(xy(s["candidates_world"][k]) - xy(v))) for s in S if k in s["candidates_world"])
    out["displaced_cards(>1.5cm)"] = [k for k, v in disp.items() if v > 0.015]; out["target_displaced"] = disp[tgt] > 0.015; out["wrong_card_displaced"] = any(v > 0.015 for k, v in disp.items() if k != tgt)
    out["target_max_lift_z"] = float(max(s["candidates_world"][tgt][2] for s in S if tgt in s["candidates_world"]) - c0[tgt][2]); out["target_is_grasped_any(native)"] = any(s["is_grasped"] for s in S)
    # D: close event
    ce = [i for i in range(1, n) if S[i]["reactive_close_event"]]
    if ce:
        i = ce[0]; d = i - 1; k, dist = nearest(S[d]["candidates_world"], ee[d]); out["close"] = {"step": i, "nearest_card_to_ee": k, "correct": k == tgt, "ee_to_gt_xy": float(np.linalg.norm(xy(ee[d]) - xy(S[d]["candidates_world"][tgt]))), "ee_nearest_dist_xy": dist, "planner_target_nearest_at_close": nearest(S[d]["candidates_world"], S[i]["planner_target_world"])[0]}
    else: out["close"] = None
    # E: fine/planner correct but EE not following
    out["n_correct_replans"] = int(sum(r["target_correct"] for r in R)); out["n_replans"] = len(R)
    # stability: monitor target per step
    mon = np.array([s["monitor_target_world"] for s in S[1:]]); ids = [nearest(S[i]["candidates_world"], S[i + 1]["monitor_target_world"])[0] for i in range(len(S) - 1)]; ftd = np.linalg.norm(np.diff(mon[:, :2], axis=0), axis=1)
    dist_ee_gt = dxy[1:]; out["monitor"] = {"switches": int(sum(a != b for a, b in zip(ids[:-1], ids[1:]))), "frac_steps_monitor_card_is_target": float(np.mean([x == tgt for x in ids])), "n": len(ids)}
    out["_ftd"] = ftd.tolist(); out["_ee_gt_xy"] = dist_ee_gt[1:].tolist(); out["_ids_target"] = [x == tgt for x in ids]; out["_same_card"] = [a == b for a, b in zip(ids[:-1], ids[1:])]
    out["planner_card_switches"] = int(sum(a["nearest_card"] != b["nearest_card"] for a, b in zip(R[:-1], R[1:]))); out["planner_correct_flags"] = [bool(r["target_correct"]) for r in R]
    return out


def main():
    R = {}
    for m in MODELS:
        f = OUT / m / "episodes_raw.json"
        if f.exists(): R[m] = [analyse(e) for e in json.load(open(f))]
    ms = list(R); print("models:", ms, "| episodes:", {m: len(R[m]) for m in ms})
    print("\n== provenance == (every episode)"); 
    for m in ms: print(f"   {m}: in_training_strings all={all(e['in_training_strings'] for e in R[m])} | sample: {R[m][0]['lang']!r}")
    print("\n== PER-EPISODE (seed target | first-plan correct/dist_xy(cm) | chunk1 toward target | min EE-GT xy(cm) | closest-approached card correct | close(step,correct) | displaced cards | success) ==")
    for i in range(len(R[ms[0]])):
        print(f" seed {R[ms[0]][i]['seed']} target {R[ms[0]][i]['target']} ({R[ms[0]][i]['n_cards']} cards)")
        for m in ms:
            e = R[m][i]; c = e["close"]; print(f"     {m:4s} plan1 {'OK ' if e['first_plan_correct'] else 'BAD'} {e['first_plan_dist_xy'] * 100:5.1f} | chunk1 {'toward' if e['first_chunk_toward_target'] else 'no    '} (cos {e['first_chunk']['cos_disp_to_gt']:+.2f} gain {e['first_chunk']['approach_gain_xy'] * 100:+5.1f}) | minXY {e['min_ee_gt_xy'] * 100:5.1f} (t={e['t_min_xy']}) | closest {'T' if e['closest_card_is_target'] else 'x'} | close {('t=%d %s' % (c['step'], 'OK' if c['correct'] else 'wrong')) if c else 'none'} | displaced {e['displaced_cards(>1.5cm)']} | succ {e['success']} | replan-correct {e['n_correct_replans']}/{e['n_replans']}")
    print("\n== SUMMARY (n episodes = %d) ==" % len(R[ms[0]]))
    def agg(m, f): return np.mean([f(e) for e in R[m]])
    rows = [("A first plan target card correct", lambda e: e["first_plan_correct"]), ("A first-plan target->GT dist xy (cm, mean)", lambda e: e["first_plan_dist_xy"] * 100), ("A first-plan dist <5cm", lambda e: e["first_plan_dist_xy"] < 0.05), ("B first 50-step chunk moves toward target", lambda e: e["first_chunk_toward_target"]),
            ("B mean approach gain over chunk 1 (cm)", lambda e: e["first_chunk"]["approach_gain_xy"] * 100), ("C min EE-GT xy (cm, mean)", lambda e: e["min_ee_gt_xy"] * 100), ("C min EE-GT xy (cm, median)", None), ("C min EE-GT 3D (cm, mean)", lambda e: e["min_ee_gt_3d"] * 100), ("C min EE-GT xy < 5cm", lambda e: e["min_ee_gt_xy"] < 0.05),
            ("  closest-approached card is target", lambda e: e["closest_card_is_target"]), ("  target card displaced (touched/pushed)", lambda e: e["target_displaced"]), ("  wrong card displaced", lambda e: e["wrong_card_displaced"]), ("D close event occurred", lambda e: e["close"] is not None),
            ("D close near correct card (of closes)", None), ("  target max lift >3cm", lambda e: e["target_max_lift_z"] > 0.03), ("  native is_grasped(target) ever", lambda e: e["target_is_grasped_any(native)"]), ("7 native task success", lambda e: e["success"]),
            ("all-replan target-correct rate (card)", lambda e: e["n_correct_replans"] / e["n_replans"]), ("planner card switches / episode", lambda e: e["planner_card_switches"])]
    print(f"   {'':46s}" + "".join(f"{m:>9s}" for m in ms))
    for name, f in rows:
        vals = []
        for m in ms:
            if name.startswith("C min EE-GT xy (cm, median)"): vals.append(np.median([e["min_ee_gt_xy"] * 100 for e in R[m]]))
            elif name.startswith("D close near"): cl = [e["close"]["correct"] for e in R[m] if e["close"]]; vals.append(np.mean(cl) if cl else float("nan"))
            else: vals.append(agg(m, f))
        print(f"   {name:46s}" + "".join(f"{v:9.3f}" for v in vals))
    print("\n== E: grounding correct at a replan, does EE follow? (chunk-level, all replans) ==")
    for m in ms:
        rr = [r for e in R[m] for r in e["replans"]]; ok = [r for r in rr if r["target_correct"]]; bad = [r for r in rr if not r["target_correct"]]
        f = lambda xs, k: float(np.mean([x[k] for x in xs])) if xs else float("nan")
        print(f"   {m:4s} replans={len(rr)} target-correct={len(ok)} ({len(ok) / len(rr):.2f}) | correct: mean approach gain {f(ok, 'approach_gain_xy') * 100:+.1f}cm, cos(disp,GT) {f(ok, 'cos_disp_to_gt'):+.2f}, frac not-approaching(gain<=0) {np.mean([r['approach_gain_xy'] <= 0 for r in ok]) if ok else float('nan'):.2f}, frac cos(disp,planner target)<0.3 {np.mean([r['cos_disp_to_planner'] < 0.3 for r in ok]) if ok else float('nan'):.2f} | wrong: gain {f(bad, 'approach_gain_xy') * 100:+.1f}cm, cos(disp,GT) {f(bad, 'cos_disp_to_gt'):+.2f}, cos(disp,planner) {f(bad, 'cos_disp_to_planner'):+.2f}")
    if "g4d" in R:
        print("\n== ANCHOR COUPLING IN CLOSED LOOP (G4d replans): CASE 1 map correct+anchor<5cm | CASE 2 map correct+anchor>5cm | CASE 3 map wrong ==")
        rr = [r for e in R["g4d"] for r in e["replans"] if "case" in r]; print(f"   replans={len(rr)}")
        for c, lab in ((1, "map correct + anchor<5cm"), (2, "map correct + anchor>5cm"), (3, "map wrong")):
            xs = [r for r in rr if r["case"] == c]
            if xs: print(f"   CASE {c} {lab:26s} n={len(xs):3d} ({len(xs) / len(rr):.2f}) | mean approach gain {np.mean([r['approach_gain_xy'] for r in xs]) * 100:+.1f}cm | frac approaching(gain>2cm) {np.mean([r['approach_gain_xy'] > 0.02 for r in xs]):.2f} | end EE-GT xy mean {np.mean([r['ee_gt_end_xy'] for r in xs]) * 100:.1f}cm | planner->GT mean {np.mean([r['planner_to_gt_xy'] for r in xs]) * 100:.1f}cm")
        print("   fine alpha location card correct (selected crop):", np.mean([r["fine_correct"] for r in rr]).round(3), "| map-card correct:", np.mean([r["map_correct"] for r in rr]).round(3), "| planner nearest-card correct:", np.mean([r["target_correct"] for r in rr]).round(3))
        for e in R["g4d"]:
            for r in e["replans"]:
                pass
        # per-episode outcome by dominant case at first plan
        print("   first-plan case per seed:", {e["seed"]: e["replans"][0].get("case") for e in R["g4d"]})
    print("\n== STABILITY (monitor-only target recomputed every frame; G4d/G3/V2 each use their own grounder) ==")
    bins = [("EE-GT xy >30cm", lambda d: d > .30), ("15-30cm", lambda d: (d > .15) & (d <= .30)), ("8-15cm", lambda d: (d > .08) & (d <= .15)), ("<8cm (late/near)", lambda d: d <= .08)]
    for m in ms:
        ftd = np.concatenate([e["_ftd"] for e in R[m]]); dd = np.concatenate([e["_ee_gt_xy"] for e in R[m]]); same = np.concatenate([e["_same_card"] for e in R[m]]); n = min(len(ftd), len(dd), len(same)); ftd, dd, same = ftd[:n], dd[:n], same[:n]
        print(f"   {m:4s} per-frame monitor-target displacement (cm) median/mean/p90: {np.median(ftd) * 100:.2f}/{ftd.mean() * 100:.2f}/{np.percentile(ftd, 90) * 100:.2f} | card switches per episode {np.mean([e['monitor']['switches'] for e in R[m]]):.1f} | frac frames monitor-card==target {np.mean([e['monitor']['frac_steps_monitor_card_is_target'] for e in R[m]]):.2f}")
        for lab, fb in bins:
            msk = fb(dd)
            if msk.sum() > 5: print(f"        {lab:18s} n={int(msk.sum()):5d}: displacement median {np.median(ftd[msk]) * 100:5.2f} mean {ftd[msk].mean() * 100:5.2f} cm | same-card-as-prev-frame {same[msk].mean():.3f}")
    json.dump({m: [{k: v for k, v in e.items() if not k.startswith("_")} for e in R[m]] for m in ms}, open(OUT / "g5_analysis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()


def extras():
    R = {m: [analyse(e) for e in json.load(open(OUT / m / "episodes_raw.json"))] for m in MODELS}
    print("\n== TIME DEPENDENCE: planner target-card correct rate by replan index (n replans) ==")
    for m in MODELS:
        out = []
        for lab, lo, hi in (("1st", 0, 1), ("2nd-3rd", 1, 3), ("4th-6th", 3, 6), ("7th+", 6, 99)):
            xs = [r["target_correct"] for e in R[m] for j, r in enumerate(e["replans"]) if lo <= j < hi]; out.append(f"{lab} {np.mean(xs):.2f} (n={len(xs)})" if xs else f"{lab} -")
        print(f"   {m:4s} " + " | ".join(out))
    print("   G4d fine-alpha card / map card correct by replan index:")
    for lab, lo, hi in (("1st", 0, 1), ("2nd-3rd", 1, 3), ("4th-6th", 3, 6), ("7th+", 6, 99)):
        xs = [r for e in R["g4d"] for j, r in enumerate(e["replans"]) if lo <= j < hi and "map_correct" in r]
        if xs: print(f"      {lab:8s} n={len(xs)} fine {np.mean([r['fine_correct'] for r in xs]):.2f} map {np.mean([r['map_correct'] for r in xs]):.2f} anchor<5cm {np.mean([r['anchor_le5'] for r in xs]):.2f}")
    print("\n== by EE-to-GT xy distance at decision time (planner card correct) ==")
    for m in MODELS:
        out = []
        for lab, lo, hi in ((">25cm", .25, 9), ("12-25cm", .12, .25), ("<12cm", 0, .12)):
            xs = [r["target_correct"] for e in R[m] for r in e["replans"] if lo <= r["ee_gt_start_xy"] < hi]; out.append(f"{lab} {np.mean(xs):.2f} (n={len(xs)})" if xs else f"{lab} -")
        print(f"   {m:4s} " + " | ".join(out))
    print("\n== E (strict): chunks with EE-GT xy > 12cm at decision (room to travel). Does the EE move toward the PLANNER target when it is correct / wrong? ==")
    for m in MODELS:
        rr = [r for e in R[m] for r in e["replans"] if r["ee_gt_start_xy"] > 0.12]; ok = [r for r in rr if r["target_correct"]]; bad = [r for r in rr if not r["target_correct"]]
        f = lambda xs, k: np.mean([x[k] for x in xs]) if xs else float("nan")
        print(f"   {m:4s} n={len(rr)} (correct {len(ok)}, wrong {len(bad)}) | correct: gain {f(ok, 'approach_gain_xy') * 100:+.1f}cm approaching(gain>2cm) {np.mean([r['approach_gain_xy'] > .02 for r in ok]) if ok else float('nan'):.2f} cos(disp,planner) {f(ok, 'cos_disp_to_planner'):+.2f} | wrong: gain {f(bad, 'approach_gain_xy') * 100:+.1f}cm cos(disp,GT) {f(bad, 'cos_disp_to_gt'):+.2f} cos(disp,planner) {f(bad, 'cos_disp_to_planner'):+.2f}")
    print("\n== first replan only: planner target correct -> what happened next (outcome per episode) ==")
    for m in MODELS:
        ok = [e for e in R[m] if e["first_plan_correct"]]; bad = [e for e in R[m] if not e["first_plan_correct"]]
        f = lambda es, k: np.mean([k(e) for e in es]) if es else float("nan")
        print(f"   {m:4s} plan1-correct n={len(ok)}: min EE-GT xy {f(ok, lambda e: e['min_ee_gt_xy']) * 100:.1f}cm, success {f(ok, lambda e: e['success']):.2f}, close-correct {sum(1 for e in ok if e['close'] and e['close']['correct'])}/{sum(1 for e in ok if e['close'])} | plan1-wrong n={len(bad)}: min EE-GT xy {f(bad, lambda e: e['min_ee_gt_xy']) * 100:.1f}cm, success {f(bad, lambda e: e['success']):.2f}")
    print("\n== sign/lengths ==")
    for m in MODELS: print(f"   {m}: success seeds {[e['seed'] for e in R[m] if e['success']]} | episode lengths {[e['len'] for e in R[m]]}")


if __name__ == "__main__":
    extras()
