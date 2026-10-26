#!/usr/bin/env python
"""G4d reference diagnostic (NO training): the saved G4c checkpoint (matcher, anchor head, value pathway all frozen) evaluated with three global-fusion rules applied to the SAME predictions (p_j, alpha_j):
 A  current multiplicative  w*(EPS+M_fine)            B  fine-only  norm(M_fine)            C  beta=0.2 convex  norm(0.8*norm(M_fine)+0.2*w)
Reports funnel stages 4/5, card selection, R_follow, selection acc, XY error (anchor head re-evaluated from the fused map; NOTE the head was trained on A-type maps)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import g4c_matcher_train as gc  # noqa: E402
from g4c_report import per_query, ab  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g4d_offline")


def main():
    OUT.mkdir(parents=True, exist_ok=True); d = gc.Data(); res = {}
    for name, fusion, beta in (("A_current_multiplicative", "mult", 0.0), ("B_fine_only", "add", 0.0), ("C_beta0.2", "add", 0.2)):
        m = gc.G4cModel(fusion, beta).to("cuda"); m.load_state_dict(torch.load(gc.OUTC / "g4c_matcher_s0.pt")); m.eval(); res[name] = {}
        for sp in ("val", "test", "test_heldout_identity"):
            e = gc.eval_sim(m, d, sp); arr = e.pop("_arrays"); q = per_query(d, arr, True); res[name][sp] = {"R_follow": e["R_follow_xy_pooled"], "selection_acc": e["selection_acc"], "median_xy_cm": e["loc_err_xy_cm_median"], "mean_xy_cm": e["loc_err_xy_cm_mean"],
                                                                                                         "map_correct_card": float(q["map"].mean()), "anchor_le_5cm": float(q["anc5"].mean()), "AB": ab(q)}
            r = res[name][sp]; print(f"{name:26s} {sp:22s} map-correct {r['map_correct_card']:.3f} | R={r['R_follow']:.3f} sel={r['selection_acc']:.3f} medXY={r['median_xy_cm']:.2f} meanXY={r['mean_xy_cm']:.2f} | <=5cm {r['anchor_le_5cm']:.3f} | A: n={r['AB']['A_correct']['n']} med {r['AB']['A_correct']['median_xy_cm']:.2f} >5cm {r['AB']['A_correct']['frac_gt_5cm']:.2f}")
    (OUT / "g4d_fusion_counterfactual.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
