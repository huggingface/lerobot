#!/usr/bin/env python
"""G4b candidate cache. Candidates = g4b_proposals.make(w) (front-view 144px crops; right only proposes via centroid->table plane->front).
w source: train scenes / train dataset episodes -> OOF fold maps (g4b_oof_maps.pt); val/test/held-out -> final G3-M1 maps (g4_cache.pt). v (value pathway), text embeddings, anchors: from g4_cache.pt (final M1; v never enters candidate selection).
Crop embedding = frozen embed_image (vision tower + connector) of the FRONT crop resized to 512, mean of its 64 tokens. GT (bboxes) -> purity diagnostics + L_fine label only."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import REPO_ID  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from g1_train import D  # noqa: E402
import g4b_proposals as P  # noqa: E402
from g4_build_cache import D0  # noqa: E402
from lerobot.policies.common.vla_utils import resize_with_pad  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

PD = D / "select_poker"; OUT = Path("outputs/eval/safediff_vla_g4b_offline"); OLD = Path("outputs/eval/safediff_vla_g4_offline")


def pack(cands, size):  # list of dicts -> box[J,3] (in `size` px), nbr[J,192], src[J], valid[J]
    box = np.zeros((P.J, 3), np.int64); nbr = np.zeros((P.J, 192), bool); src = np.zeros(P.J, np.int64); valid = np.zeros(P.J, bool)
    for j, c in enumerate(cands):
        x0, y0, s = c["box"]; f = size / 480; box[j] = (int(round(x0 * f)), int(round(y0 * f)), max(2, int(round(s * f)))); nbr[j] = c["nbr"]; src[j] = c["src"] == "right"; valid[j] = True
    return box, nbr, src, valid


def main():
    old = torch.load(OLD / "g4_cache.pt", weights_only=False); oof = torch.load(OUT / "g4b_oof_maps.pt", weights_only=False); s = old["sim"]; S = len(s["sids"]); valid = s["valid"].numpy(); split = np.array(s["split"])
    B = {}
    for p in sorted(PD.glob("bboxes_*.json")): B.update(json.load(open(p)))
    keys = {}

    def kid(k): return keys.setdefault(k, len(keys))

    # ---- sim ----
    wsim = s["w"].float().numpy().copy(); is_oof = split == "train"; wsim[is_oof] = oof["w_sim"].numpy()[is_oof]; assert not np.isnan(wsim[is_oof][valid[is_oof]]).any()
    sim = {k: np.zeros(sh, dt) for k, sh, dt in (("box", (S, 4, P.J, 3), np.int64), ("nbr", (S, 4, P.J, 192), bool), ("src", (S, 4, P.J), np.int64), ("valid", (S, 4, P.J), bool), ("crop", (S, 4, P.J), np.int64))}
    pur = {k: np.zeros((S, 4, P.J), np.float32) for k in ("contains_target", "n_cards", "target_nearest", "tgt_frac", "dis_frac")}
    for si in range(S):
        sid = s["sids"][si]; names = s["names"][si]; gtb = [B[sid]["boxes_xyxy_per_camera"][names[c]][0] if (c < len(names) and valid[si, c]) else None for c in range(len(names))]
        for k in range(4):
            if not valid[si, k]: continue
            cands = P.make(wsim[si, k].astype(np.float64)); b, n, sr, v = pack(cands, 480); sim["box"][si, k], sim["nbr"][si, k], sim["src"][si, k], sim["valid"][si, k] = b, n, sr, v
            for j, c in enumerate(cands):
                sim["crop"][si, k, j] = kid(("sim", sid, *map(int, b[j])))
                for kk, val in P.purity(c["box"], gtb, k).items(): pur[kk][si, k, j] = val
    # ---- dataset ----
    ds = {}
    for name in ("dtrain", "dvald", "dheld"):
        d = old[name]; n, V = d["cell"].shape[:2]; w = d["w"].float().numpy().copy()
        if name == "dtrain": w = oof["w_dtrain"].numpy().copy(); assert not np.isnan(w).any()
        e = {k: np.zeros(sh, dt) for k, sh, dt in (("box", (n, V, P.J, 3), np.int64), ("nbr", (n, V, P.J, 192), bool), ("src", (n, V, P.J), np.int64), ("valid", (n, V, P.J), bool), ("crop", (n, V, P.J), np.int64))}
        for i in range(n):
            for t in range(V):
                cands = P.make(w[i, t].astype(np.float64)); b, nb, sr, v = pack(cands, 224)
                for k_, val in zip(("box", "nbr", "src", "valid"), (b, nb, sr, v)): e[k_][i, t] = val
                for j in range(len(cands)): e["crop"][i, t, j] = kid((name, i, *map(int, b[j])))
        ds[name] = {"w": torch.from_numpy(w), **e}; print(name, "candidates", n, V, flush=True)
    print("unique crops", len(keys), flush=True)
    # ---- encode ----
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); vlm = policy.backbone.model.vlm_with_expert; scale = math.sqrt(960)
    dset = LeRobotDataset(REPO_ID); from_idx = dset.meta.episodes["dataset_from_index"]; cam0 = dset.meta.camera_keys[0]; eps = pd.read_json(D0 / "episodes.json").set_index("episode"); by_img = {}
    for key, idx in keys.items(): by_img.setdefault((key[0], key[1]), []).append((key, idx))
    emb = torch.zeros(len(keys), 960, dtype=torch.float16)

    @torch.no_grad()
    def embed(imgs):
        x = torch.cat([resize_with_pad(torch.from_numpy(i).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0, 512, 512, pad_value=0) * 2.0 - 1.0 for i in imgs]); return (vlm.embed_image(x).float() * scale).mean(1).cpu()

    for n_, ((dom, img), lst) in enumerate(by_img.items()):
        if dom == "sim": fr = np.load(PD / f"{img}.npz")["rgb"][0]
        else:
            m = old[dom]["meta"][img]; f = int(eps.loc[m["episode"], "t_close"]) - int(m["k"]) if dom == "dtrain" else int(m["image_frame"]); item = dset[int(from_idx[int(m["episode"])]) + f]; fr = item[cam0].permute(1, 2, 0).numpy(); fr = (fr * 255).astype(np.uint8) if fr.dtype != np.uint8 else fr
        crops = [fr[k[3]:k[3] + k[4], k[2]:k[2] + k[4]] for k, _ in lst]; e = embed(crops)
        for (k, idx), a in zip(lst, e): emb[idx] = a.half()
        if (n_ + 1) % 500 == 0: print("images", n_ + 1, "/", len(by_img), flush=True)
    cache = {"sim": {**sim, "w": torch.from_numpy(wsim), "purity": pur, "fold": oof["sim_fold"]}, **ds, "crops": {"emb": emb, "keys": [None] * len(keys)}}
    for k, i in keys.items(): cache["crops"]["keys"][i] = k
    torch.save(cache, OUT / "g4b_cache.pt"); print("saved")


if __name__ == "__main__":
    main()
