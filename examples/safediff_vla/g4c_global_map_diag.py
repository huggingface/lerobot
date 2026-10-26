#!/usr/bin/env python
"""Diagnostic only (no training, no change to the model): where does correct-card information get lost between the intra-crop alpha and the final global map?
Variants of the card-selection read-out (card = argmax_c sum(map * cellcover_c)), restricted to queries where the fine-selected crop contains the target:
  w2        : trained final map (w * (EPS + sum_j p_j M_j))                      [what the anchor head sees]
  fine_only : sum_j p_j M_j  (no multiplication by the coarse map)
  w_coarse  : coarse map only
  alpha_pix : card with the largest alpha mass inside its GT token set in the selected crop (= funnel stage 3)
Also: fraction of target cards whose 60px coarse cell(s) are shared with another card (cell-resolution ceiling), and stage-4 accuracy split by whether the coarse-map argmax card was the target."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import g4c_matcher_train as gc  # noqa: E402

OC = gc.OUTC


def main():
    d = gc.Data(); model = gc.G4cModel().to("cuda"); model.load_state_dict(torch.load(OC / "g4c_matcher_s0.pt")); model.eval(); A = torch.load(OC / "g4c_matcher_s0_arrays.pt", weights_only=False); cf = d.sim["cellfrac"].numpy(); valid = d.sim["valid"].numpy(); res = {}
    for sp in ("val", "test", "test_heldout_identity"):
        a = A[sp]; idx = a["idx"]; t = torch.as_tensor(idx, device="cuda"); cv = d.cvalid[t].cpu().numpy(); cont = d.pur["contains_target"][t].cpu().numpy() > .5; top = np.where(cv, a["s"], -1e9).argmax(1); r = np.arange(len(idx)); sel = cont[r, top]
        Mj = np.einsum("njt,njtc->njc", a["alpha"], d.proj[t].float().cpu().numpy()); fine_only = (a["p"][:, :, None] * Mj).sum(1); w0 = d.w[t].cpu().numpy(); k = d.pairs[idx, 1]
        def correct(m): return np.array([max([c for c in range(4) if valid[d.pairs[i, 0], c]], key=lambda c: float((m[n] * cf[d.pairs[i, 0], c]).sum())) == k[n] for n, i in enumerate(idx)])
        c_w2, c_fine, c_w0 = correct(a["w2"]), correct(fine_only), correct(w0)
        shared = []
        for n, i in enumerate(idx):
            sc = d.pairs[i, 0]; tc = cf[sc, k[n]] > 0; oth = sum(cf[sc, c] for c in range(4) if valid[sc, c] and c != k[n]) > 0; shared.append(bool((tc & oth).any()))
        shared = np.array(shared); o = {"n_selected_crop_has_target": int(sel.sum()), "map_correct_given_selected_crop_has_target": {"w2": float(c_w2[sel].mean()), "fine_only": float(c_fine[sel].mean()), "coarse_w": float(c_w0[sel].mean())},
                                           "coarse_argmax_card_was_target": {"frac": float(c_w0[sel].mean()), "w2_correct_if_coarse_correct": float(c_w2[sel & c_w0].mean()), "w2_correct_if_coarse_wrong": float(c_w2[sel & ~c_w0].mean()), "n_coarse_wrong": int((sel & ~c_w0).sum())},
                                           "target_cell_shared_with_another_card_frac": float(shared.mean()), "w2_correct_if_target_cell_shared": float(c_w2[sel & shared].mean()) if (sel & shared).any() else None, "w2_correct_if_not_shared": float(c_w2[sel & ~shared].mean())}
        res[sp] = o; print(sp, json.dumps(o))
    (OC / "g4c_global_map_diag.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
