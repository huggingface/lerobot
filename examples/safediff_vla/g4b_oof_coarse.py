#!/usr/bin/env python
"""G4b OOF proposal maps: scene-level 4-fold. Fold f's coarse model = the unchanged G3-M1 recipe (g2_train.run, mix M1, seed 0, 2000 steps, eval/100, same VAL-based selection) trained WITHOUT fold-f
train scenes (all 4 referent queries of a scene share the fold) and WITHOUT fold-f dataset-demonstration episodes; it then produces the coarse maps w for the held fold. val/test/held-out are untouched."""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g2_train as G2  # noqa: E402
from g1_train import Bottleneck, TaskData, V2Modules  # noqa: E402
from g4_build_cache import frozen_coarse  # noqa: E402

E3 = Path("outputs/eval/safediff_vla_g3_offline"); OUT = Path("outputs/eval/safediff_vla_g4b_offline"); OUTT = Path("outputs/train/safediff_vla_g4b_oof")


def main():
    dev = "cuda"; sim = TaskData("select_poker", dev); dtrain = G2.DataCache(E3 / "g3_train_dataset_cache.pt", dev); dval = G2.DataCache(E3 / "g3_vald_cache.pt", dev); dheld = G2.DataCache(E3 / "g3_held_cache.pt", dev)
    refs = {"vald": G2.eval_frames(V2Modules().to(dev), dval, sim.mean, sim.std, False)}; rng = np.random.default_rng(2024)
    tr = sim.idx("train"); sfold = {int(s): i % 4 for i, s in enumerate(rng.permutation(tr))}
    eps = sorted({m["episode"] for m in dtrain.meta}); efold = {e: i % 4 for i, e in enumerate(rng.permutation(eps))}; dfold = np.array([efold[m["episode"]] for m in dtrain.meta])
    w_sim = torch.full((len(sim.sids), 4, 192), float("nan")); w_dtr = torch.full((len(dtrain.meta), 1, 192), float("nan")); log = {"scene_fold": {sim.sids[s]: f for s, f in sfold.items()}, "folds": {}}
    for f in range(4):
        sim2 = copy.copy(sim); keep = np.array([s for s in tr if sfold[int(s)] != f]); hold = np.array([s for s in tr if sfold[int(s)] == f]); orig = sim.idx; sim2.idx = lambda sp, _o=orig, _k=keep: _k if sp == "train" else _o(sp)
        km = np.where(dfold != f)[0]; d2 = copy.copy(dtrain); d2.tok = dtrain.tok[km]; d2.mod = dtrain.mod[km]; d2.anc = dtrain.anc[km]; d2.meta = [dtrain.meta[i] for i in km]
        G2.OUTT = OUTT / f"fold{f}"; print(f"=== fold {f}: train scenes {len(keep)} (held {len(hold)}), dataset samples {len(km)} (held {int((dfold == f).sum())})", flush=True)
        o = G2.run("M1", 0, 2000, 100, sim2, d2, dval, dheld, refs, None); t = o["test"]
        print(f"fold {f} coarse: step={o['selected_step']} guardOK={o['guard_satisfied_at_selected_on_valD']} | test R={t['R_follow_xy_pooled']:+.3f} acc={t['selection_acc']:.2f}", flush=True)
        m = Bottleneck().to(dev); m.load_state_dict(torch.load(G2.OUTT / "M1_s0.pt")); m.eval()
        ts = sim.tok[torch.as_tensor(hold, device=dev)]; n = len(hold); w, _ = frozen_coarse(m, ts.reshape(n * 4, *ts.shape[2:]), sim.mod[torch.as_tensor(hold, device=dev)].reshape(n * 4, -1)); w_sim[hold] = w.reshape(n, 4, 192).float().cpu()
        hd = np.where(dfold == f)[0]; w, _ = frozen_coarse(m, dtrain.tok[hd, 0], dtrain.mod[hd, 0]); w_dtr[hd, 0] = w.float().cpu()
        log["folds"][f] = {"selected_step": o["selected_step"], "val_R": o["val"]["R_follow_xy_pooled"], "val_acc": o["val"]["selection_acc"], "test_R": t["R_follow_xy_pooled"], "test_acc": t["selection_acc"], "guard_ok": o["guard_satisfied_at_selected_on_valD"], "n_train_scenes": len(keep), "n_held_scenes": len(hold)}
    torch.save({"w_sim": w_sim, "w_dtrain": w_dtr, "sim_fold": sfold, "dtrain_fold": dfold}, OUT / "g4b_oof_maps.pt"); (OUT / "g4b_oof_log.json").write_text(json.dumps(log, indent=1)); print("saved")


if __name__ == "__main__":
    main()
