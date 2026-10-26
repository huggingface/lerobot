#!/usr/bin/env python
"""Diagnostic: after the fine matcher picks a crop that contains the target, does the FINAL map select the correct CARD? Stratified by #cards in the chosen (positive) crop. Test + held-out."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import g4b_matcher_train as g

OUT = Path("outputs/eval/safediff_vla_g4b_offline")


def main():
    d = g.Data(); A = torch.load(OUT / "g4b_matcher_s0_arrays.pt", weights_only=False); cf = d.sim["cellfrac"].numpy(); valid = d.sim["valid"].numpy(); res = {}
    for sp in ("val", "test", "test_heldout_identity"):
        a = A[sp]; idx = a["idx"]; t = torch.as_tensor(idx, device="cuda"); cv = d.cvalid[t].cpu().numpy(); cont = d.pur["contains_target"][t].cpu().numpy() > .5; nc = d.pur["n_cards"][t].cpu().numpy(); near = d.pur["target_nearest"][t].cpu().numpy() > .5
        top = np.where(cv, a["s"], -1e9).argmax(1); r = np.arange(len(idx)); ok = cont[r, top]; rows = []
        for n, i in enumerate(idx):
            sc, k = d.pairs[i]; ks = [c for c in range(4) if valid[sc, c]]; sel = max(ks, key=lambda c: float((a["w2"][n] * cf[sc, c]).sum())); rows.append(sel == k)
        rows = np.array(rows); nct = nc[r, top]; o = {"n": len(idx), "fine_top1_crop_contains_target": float(ok.mean()), "map_selects_correct_card_overall": float(rows.mean()), "map_correct_given_top1_crop_correct": float(rows[ok].mean()), "map_correct_given_top1_crop_wrong": float(rows[~ok].mean())}
        for lab, m in (("1_card_in_chosen_crop", ok & (nct == 1)), ("2plus_cards_in_chosen_crop", ok & (nct >= 2)), ("target_is_centre_card_of_chosen_crop", ok & near[r, top]), ("target_not_centre_card", ok & ~near[r, top])): o[lab] = {"n": int(m.sum()), "map_correct": float(rows[m].mean()) if m.any() else None}
        res[sp] = o; print(sp, json.dumps(o))
    (OUT / "g4b_intracrop_diag.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
