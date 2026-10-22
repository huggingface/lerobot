#!/usr/bin/env python
"""[v2.1HR-expansion] SAME HR scorer (v21hr_candidate.Scorer variant B, same recipe: seed 1000, 3000 steps, batch 32, AdamW 3e-4/1e-2, final step) on the expanded
poker/mahjong scenes (200 train / 50 val per task). No architecture change. Extra evaluations:
  (i)  data-size control: the OLD 40-scene-trained scorer evaluated on the NEW 50-scene val sets;
  (ii) identity stratification of val targets by train frequency (seen>=10 / rare 1-9 / unseen 0);
  (iii) unseen-identity probe (ONE extra run): K identities are removed from the training candidate sets (their crops AND instructions never seen), evaluated on val
        samples whose target is a removed identity (full candidate sets).
Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/v21x_hr.py"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import v21hr_candidate as H  # noqa: E402
from v21_common import OUT as V21, load_policy, make_batch  # noqa: E402

X = V21 / "x"; TASKS = H.TASKS


@torch.no_grad()
def text_tokens(pol, pre, rows):
    T = {}
    for r in rows:
        if r["task"] not in TASKS or not r["scene_ok"]: continue
        for ci, c in enumerate(r["candidates"]):
            if not c["valid"]: continue
            tok, pad, mod = pol._encode_multimodal_latent(pre(make_batch(r, c["instruction"]))); m = (mod[0] == 1) & pad[0].bool(); T[(r["scene_id"], ci)] = tok[0][m].float().cpu()
    return T


def samples(rows, split, ent, T, drop_ids=None):
    S = []
    for r in rows:
        if r["task"] not in TASKS or not r["scene_ok"]: continue
        ids = [i for i, c in enumerate(r["candidates"]) if c["valid"] and not (drop_ids and r["candidates"][i]["name"] in drop_ids)]
        if len(ids) < 2: continue
        for ti in ids:
            S.append({"scene": r["scene_id"], "task": r["task"], "split": split[r["scene_id"]], "ids": ids, "t": ids.index(ti), "tname": r["candidates"][ti]["name"], "txt": T[(r["scene_id"], ti)], "B": torch.stack([ent[(r["scene_id"], i)]["toks"].float() for i in ids])})
    return S


def train(Str, seed=H.SEED, tag=""):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = H.Scorer("B").cuda(); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-2)
    for step in range(1, H.STEPS + 1):
        m.train(); idx = rng.choice(len(Str), H.BATCH, replace=False); txt, tm, f, cm, tgt = H.collate([Str[i] for i in idx], "B"); loss = F.cross_entropy(m(txt, tm, f).masked_fill(~cm, -1e9), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0: print(tag, step, float(loss), flush=True)
    return m


def geo(out, rows_by, off):
    res = {}
    for t in TASKS:
        by = {}
        for o in out:
            if o["task"] == t: by.setdefault(o["scene"], {})[o["t"]] = o
        cos, real = [], []
        for sc, d in by.items():
            r = rows_by[sc]; A = np.stack([r["candidates"][i]["anchor"] for i in d[next(iter(d))]["ids"]]) + off[t]
            for a in d:
                for b in d:
                    if a == b: continue
                    p = A[d[b]["p"].argmax()] - A[d[a]["p"].argmax()]; g = A[b] - A[a]; ng = np.linalg.norm(g); np_ = np.linalg.norm(p); cos.append(g @ p / (ng * np_) if np_ > 1e-9 else 0.0); real.append(p @ g / ng ** 2)
        if cos: res[t] = {"cos": float(np.mean(cos)), "realized": float(np.mean(real)), "pairs": len(cos)}
    return res


def strat(out, S_by_key, counts):
    res = {}
    for t in TASKS:
        for name, lo, hi in (("seen>=10", 10, 10 ** 9), ("rare 1-9", 1, 9), ("unseen 0", 0, 0)):
            O = [o for o in out if o["task"] == t and lo <= counts[t].get(S_by_key[(o["scene"], o["t"], tuple(o["ids"]))], 0) <= hi]
            if O: res[f"{t}|{name}"] = {"n": len(O), "acc": float(np.mean([o["p"].argmax() == o["t"] for o in O])), "chance": float(np.mean([1 / o["n"] for o in O]))}
    return res


def main():
    rows = pickle.load(open(X / "referent_table_x.pkl", "rb")); split = json.load(open(X / "split_x.json")); rb = {r["scene_id"]: r for r in rows}
    from v21_common import offsets; off = offsets()
    pol, pre = load_policy(); ent = H.extract(pol, rows); T = text_tokens(pol, pre, rows); S = samples(rows, split, ent, T); tr = [s for s in S if s["split"] == "train"]; va = [s for s in S if s["split"] == "val"]
    print("expanded train samples", len(tr), "val samples", len(va), flush=True); res = {}
    # identity counts (train candidate appearances)
    counts = {t: {} for t in TASKS}
    for r in rows:
        if split[r["scene_id"]] == "train":
            for c in r["candidates"]:
                if c["valid"]: counts[r["task"]][c["name"]] = counts[r["task"]].get(c["name"], 0) + 1
    key = {(s["scene"], s["t"], tuple(s["ids"])): s["tname"] for s in S}
    m = train(tr, tag="expanded"); otr, ova = H.evaluate(m, tr, "B"), H.evaluate(m, va, "B")
    res["expanded"] = {"train": H.metrics(otr, "train", rb), "val": H.metrics(ova, "val", rb), "geometric_swap_val": geo(ova, rb, off), "val_by_identity_frequency": strat(ova, key, counts), "train_identity_counts": {t: sorted(c.values()) for t, c in counts.items()}}
    pickle.dump(ova, open(X / "val_scores_expanded.pkl", "wb")); torch.save(m.state_dict(), X / "scorer_expanded.pt")
    # (i) old 40-scene scorer on new val
    orows = pickle.load(open(V21 / "referent_table.pkl", "rb")); osplit = json.load(open(V21 / "pilot_scene_split.json")); ocache = torch.load(V21 / "pilot_token_cache.pt", weights_only=False)
    oent = H.extract(pol, orows); OS = H.build_samples(orows, osplit, ocache, oent); otr_s = [s for s in OS if s["split"] == "train"]
    mo = train(otr_s, tag="old40"); ov = H.evaluate(mo, va, "B"); res["old40_trained_on_new_val"] = {"val": H.metrics(ov, "val", rb), "geometric_swap_val": geo(ov, rb, off)}
    # (iii) unseen-identity probe
    rng = np.random.default_rng(5); hold = {}
    for t, k in (("select_poker", 8), ("select_mahjong", 3)): ids = sorted(counts[t]); hold[t] = set(rng.choice(ids, size=k, replace=False).tolist())
    drop = set().union(*hold.values()); S2 = samples(rows, split, ent, T, drop_ids=drop); tr2 = [s for s in S2 if s["split"] == "train"]; mp = train(tr2, tag="holdout")
    vh = [s for s in S if s["split"] == "val" and s["tname"] in hold[s["task"]]]; oh = H.evaluate(mp, vh, "B"); oseen = H.evaluate(mp, [s for s in S if s["split"] == "val" and s["tname"] not in hold[s["task"]]], "B")
    res["unseen_identity_probe"] = {"held_out_identities": {t: sorted(v) for t, v in hold.items()}, "val_unseen_targets": H.metrics(oh, "val", rb), "val_seen_targets": H.metrics(oseen, "val", rb), "n_train_samples": len(tr2)}
    json.dump(res, open(X / "hr_expanded_results.json", "w"), indent=1, default=float)
    f = lambda d: {t: (round(v["acc"], 3), round(v["chance"], 3), round(v["top2"], 3), round(v["margin"], 3), round(v["swap_both_correct"], 3), round(v["pred_changes_on_swap"], 3), v["n"]) for t, v in d.items()}
    print("expanded TRAIN", f(res["expanded"]["train"])); print("expanded VAL (acc,chance,top2,margin,swap_ok,pred_changes,n)", f(res["expanded"]["val"])); print("geo", res["expanded"]["geometric_swap_val"])
    print("old40 on new val", f(res["old40_trained_on_new_val"]["val"])); print("strat", res["expanded"]["val_by_identity_frequency"])
    print("unseen probe held", res["unseen_identity_probe"]["held_out_identities"]); print(" unseen targets", f(res["unseen_identity_probe"]["val_unseen_targets"])); print(" seen targets", f(res["unseen_identity_probe"]["val_seen_targets"]))

if __name__ == "__main__": main()
