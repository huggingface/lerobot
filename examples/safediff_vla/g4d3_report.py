#!/usr/bin/env python
"""G4d 3-seed reproducibility + instruction-swap-sensitivity audit. Reports per seed: map-correct, R_follow(+CI), selection, median XY, anchor<=5cm, funnel, anchor A/B, retention (median+mean XY), swap shift (per frame: mean/median/frac>5cm/frac>30cm).
Reference rows: G3 M1, G4b, G4c (seed 0), V2 (untouched)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g4c_matcher_train as gc  # noqa: E402
import g4b_matcher_train as gb  # noqa: E402
from g1_train import V2Modules, TaskData  # noqa: E402
from g2_train import DataCache, eval_frames, FRAMES  # noqa: E402
from g4c_report import per_query, funnel, ab  # noqa: E402

OB = Path("outputs/eval/safediff_vla_g4b_offline"); OC = gc.OUTC; OD = Path("outputs/eval/safediff_vla_g4d_offline"); G3 = Path("outputs/eval/safediff_vla_g3_offline"); SEEDS = (0, 1, 2); SPL = ("val", "test", "test_heldout_identity")


@torch.no_grad()
def swap_stats(model, d, name="dheld"):
    """per-frame swap shift: for every held-out dataset sample, mean over the 2 swapped referents of ||pred(swap)-pred(true)||_xy (cm); also original-instruction error."""
    model.eval(); n = d.d[name]["n"]; P = []
    for t in range(3):
        pr = []
        for i in range(0, n, 64): a = d.fwd_data(model, name, np.arange(i, min(n, i + 64)), t)[0]; pr.append(gc.to_m(a, d).cpu())
        P.append(torch.cat(pr).numpy())
    fr = np.array([m["frame"] for m in d.d[name]["meta"]]); sh = np.stack([np.linalg.norm((P[t] - P[0])[:, :2], axis=1) * 100 for t in (1, 2)], 1); anc = d.d[name]["anc"].cpu().numpy(); e = np.linalg.norm((P[0] - anc)[:, :2], axis=1) * 100; out = {}
    for k in FRAMES:
        m = fr == k; s = sh[m].mean(1); out[k] = {"shift_mean": float(s.mean()), "shift_median": float(np.median(s)), "frac_shift_gt5": float((s > 5).mean()), "frac_shift_gt30": float((s > 30).mean()), "orig_med_err": float(np.median(e[m])), "orig_mean_err": float(e[m].mean())}
    return out, P


def main():
    d = gc.Data(); R = {}; sd = {s: json.load(open(OD / f"g4d_smoke_s{s}.json")) for s in SEEDS}; arr = {s: torch.load(OD / f"g4d_matcher_s{s}_arrays.pt", weights_only=False) for s in SEEDS}
    r3 = json.load(open(G3 / "g3_M1_s0.json")); r4b = json.load(open(OB / "g4b_smoke_s0.json")); r4c = json.load(open(OC / "g4c_smoke_s0.json")); A4b = torch.load(OB / "g4b_matcher_s0_arrays.pt", weights_only=False); A4c = torch.load(OC / "g4c_matcher_s0_arrays.pt", weights_only=False)
    rows = {"G4b": (r4b, A4b, False), "G4c": (r4c, A4c, True), **{f"G4d s{s}": (sd[s], arr[s], True) for s in SEEDS}}
    # G3 anchor<=5cm from its saved predictions
    sim = TaskData("select_poker", "cpu"); Z = torch.load("outputs/train/safediff_vla_g3/M1_s0_preds.pt", weights_only=False)
    def g3_le5(sp):
        P, W = Z["preds"][sp]; ids = sim.idx(sp); e = []
        for j, s in enumerate(ids):
            for k in range(4):
                if sim.valid[s, k] and bool(sim.held[s, k]) == (sp == "test_heldout_identity"): e.append(np.linalg.norm((P[j, k].numpy() - sim.anc[s, k].numpy())[:2]) * 100)
        return float((np.array(e) < 5).mean())
    print("== PRIMARY TABLE ==")
    for sp in ("test", "test_heldout_identity", "val"):
        print(f"\n[{sp}]  map-correct | R_follow [CI] | selection | median XY cm | anchor<=5cm")
        vals = {}
        x = r3[sp] if sp in r3 else None
        if x is not None: print(f"  {'G3 M1':9s}      -     | {x['R_follow_xy_pooled']:.3f} {x['R_follow_ci95_scene_bootstrap']} | {x['selection_acc']:.3f} | {x['loc_err_xy_cm_median']:.2f} | {g3_le5(sp):.3f}")
        for n, (rr, A, wa) in rows.items():
            q = per_query(d, A[sp], wa); x = rr[sp]; vals[n] = {"map": float(q["map"].mean()), "R": x["R_follow_xy_pooled"], "sel": x["selection_acc"], "med": x["loc_err_xy_cm_median"], "le5": float(q["anc5"].mean()), "mean": x["loc_err_xy_cm_mean"]}
            R.setdefault(sp, {})[n] = {**vals[n], "CI": x["R_follow_ci95_scene_bootstrap"], "funnel": funnel(q), "AB": ab(q)}
            print(f"  {n:9s} {vals[n]['map']:.3f}     | {x['R_follow_xy_pooled']:.3f} {x['R_follow_ci95_scene_bootstrap']} | {x['selection_acc']:.3f} | {x['loc_err_xy_cm_median']:.2f} | {vals[n]['le5']:.3f}")
        g = [vals[f"G4d s{s}"] for s in SEEDS]; print("  G4d mean[min,max]: " + " | ".join(f"{k} {np.mean([v[k] for v in g]):.3f} [{min(v[k] for v in g):.3f},{max(v[k] for v in g):.3f}]" for k in ("map", "R", "sel", "med", "le5")))
    print("\n== FUNNEL (test) ==  rows: stage; cols: G4c, G4d s0, s1, s2")
    names = ["G4c", "G4d s0", "G4d s1", "G4d s2"]; ks = ["1_proposal_contains_target", "2_fine_top1_crop_correct", "3_intra_card_correct(uncond)", "4_final_map_correct_card(uncond)", "5_anchor<5cm(uncond)", "5_cum"]
    for sp in ("test", "test_heldout_identity"):
        print(sp)
        for k in ks: print(f"   {k:34s}" + "".join(f"{R[sp][n]['funnel'][k]:8.3f}" for n in names))
        print(f"   {'stage3->4 drop':34s}" + "".join(f"{R[sp][n]['funnel']['3_intra_card_correct(uncond)'] - R[sp][n]['funnel']['4_final_map_correct_card(uncond)']:8.3f}" for n in names))
    print("\n== ANCHOR COUPLING: map-correct subset ==  (n, median, mean, frac>5cm)")
    for sp in ("test", "test_heldout_identity"):
        for n in ["G4b", "G4c", "G4d s0", "G4d s1", "G4d s2"]: a = R[sp][n]["AB"]["A_correct"]; print(f"   {sp:22s} {n:7s} n={a['n']:3d} med {a['median_xy_cm']:.2f} mean {a['mean_xy_cm']:.2f} >5cm {a['frac_gt_5cm']:.2f}")
    print("\n== INTRA-CROP localization (test, positive crops: top-token / mass / card) ==")
    for n, rr in (("G4c", r4c), *((f"G4d s{s}", sd[s]) for s in SEEDS)): x = rr["test"]["intra"]["positive_crops"]; y = rr["test_heldout_identity"]["intra"]["positive_crops"]; print(f"   {n:7s} test {x['top_token_in_target']:.3f}/{x['target_mass']:.3f}/{x['correct_card_multi']:.3f} | rank9 {y['top_token_in_target']:.3f}/{y['target_mass']:.3f}/{y['correct_card_multi']:.3f}")
    # ---------------- retention + swap ----------------
    refm = V2Modules().to("cuda"); dheld = DataCache(G3 / "g3_held_cache.pt", "cuda"); v2 = eval_frames(refm, dheld, d.mean, d.std, True); fr = np.array([m["frame"] for m in dheld.meta]); keys = ("f0", "k45", "k30", "k10", "pooled")
    print("\n== RETENTION: held-out dataset episodes, original instruction: median / mean XY (cm) ==")
    print(f"{'':8s}" + "".join(f"{k:>14s}" for k in keys))
    def fmt(rr): return "".join(f"{rr[k]['median_xy_cm']:7.2f}/{rr[k]['mean_xy_cm']:5.2f}" for k in keys)
    print(f"{'V2':8s}" + "".join(f"{v2[k]['median_xy_cm']:7.2f}/{v2[k]['mean_xy_cm']:5.2f}" for k in keys)); print(f"{'G3 M1':8s}" + fmt(r3["retention_heldout"])); print(f"{'G4c':8s}" + fmt(r4c["retention_heldout"]))
    for s in SEEDS: print(f"{'G4d s' + str(s):8s}" + fmt(sd[s]["retention_heldout"]) + f"   guard_ok={sd[s]['guard_ok_at_selected']} step={sd[s]['selected_step']}")
    print("\n== INSTRUCTION-SWAP SENSITIVITY (mean over 2 swapped referents of ||pred(swap)-pred(true)||_xy, cm) ==  columns: f0 k45 k30 k10 | [median, frac>5cm, frac>30cm]")
    sw = {}; mc = gc.G4cModel("mult").to("cuda"); mc.load_state_dict(torch.load(OC / "g4c_matcher_s0.pt")); sw["G4c"], _ = swap_stats(mc, d)
    for s in SEEDS: m = gc.G4cModel("add", 0.2).to("cuda"); m.load_state_dict(torch.load(OD / f"g4d_matcher_s{s}.pt")); sw[f"G4d s{s}"], _ = swap_stats(m, d)
    v2s = v2["swap_sensitivity"]; sw["V2"] = {k: {"shift_mean": v2s[k]["pred_shift_xy_cm_mean"], "shift_median": v2s[k]["pred_shift_xy_cm_median"], "frac_shift_gt5": None, "frac_shift_gt30": v2s[k]["frac_shift_gt_30cm"]} for k in FRAMES}
    g3s = r3["retention_heldout"]["swap_sensitivity"]; sw["G3 M1"] = {k: {"shift_mean": g3s[k]["pred_shift_xy_cm_mean"], "shift_median": g3s[k]["pred_shift_xy_cm_median"], "frac_shift_gt5": None, "frac_shift_gt30": g3s[k]["frac_shift_gt_30cm"]} for k in FRAMES}
    for n in ("V2", "G3 M1", "G4c", "G4d s0", "G4d s1", "G4d s2"):
        x = sw[n]; print(f"   {n:7s} mean  " + " ".join(f"{x[k]['shift_mean']:6.2f}" for k in FRAMES) + " | median " + " ".join(f"{x[k]['shift_median']:6.2f}" for k in FRAMES) + " | >5cm " + " ".join((f"{x[k]['frac_shift_gt5']:.2f}" if x[k]['frac_shift_gt5'] is not None else " n/a ") for k in FRAMES) + " | >30cm " + " ".join(f"{x[k]['frac_shift_gt30']:.2f}" for k in FRAMES))
    print("   (json check) swap_shift_xy_cm: " + " ; ".join(f"s{s} " + str({k: round(v, 2) for k, v in sd[s]['retention_heldout']['swap_shift_xy_cm'].items()}) for s in SEEDS))
    json.dump({"primary": R, "swap": sw}, open(OD / "g4d3_report.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
