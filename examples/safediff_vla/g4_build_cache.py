#!/usr/bin/env python
"""G4 crop cache + frozen coarse-path cache (forward only; no training, no weights changed).
Frozen: G3-M1 seed-0 coarse path (query extractor, referent scorer Wk/Wq/Wv/bcam, LayerNorms), frozen VLM vision tower + connector.
Per (sample = scene/frame x query): coarse map w [192]; frozen value features v = Wv(LN(image tokens)) [192,256]; candidate crops = per static camera (front, right)
top-2 peaks of the coarse 8x8 map with 3x3 NMS (<=4 crops), crop = fixed-size square at the peak CELL CENTRE (front 96px / right 136px at 480px; the same fraction of the
image for 224px dataset frames). GT (bboxes / segmentation) is used ONLY for purity diagnostics and the L_fine positive label; it never moves or removes a crop.
Crop embedding = frozen embed_image (vision tower + connector) of the crop resized to 512, mean over its 64 tokens (+ centre-4x4 mean).
Purity diagnostics (sim only): crop centre, intended GT card centre, centre distance, #cards inside, target/distractor area fractions, target-nearest-to-centre."""
from __future__ import annotations

import copy, json, math, sys
from pathlib import Path

import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, REPO_ID  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from g1_train import D, Bottleneck, TaskData  # noqa: E402

from lerobot.envs import make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.common.vla_utils import resize_with_pad  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

PD = D / "select_poker"; G3E = Path("outputs/eval/safediff_vla_g3_offline"); OUT = Path("outputs/eval/safediff_vla_g4_offline"); COARSE = "outputs/train/safediff_vla_g3/M1_s0.pt"
SIDE480 = {0: 96, 1: 136}; D0 = Path("outputs/eval/safediff_vla_phase_d0_training_data_audit")


def nms_peaks(m, K):
    m = m.copy(); out = []
    for _ in range(K):
        r, c = np.unravel_index(int(m.argmax()), m.shape); out.append(r * 8 + c); m[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = -1e9
    return out


def crop_geom(cell, cam, size):
    """crop box (x0,y0,side) in an image of `size` px for a coarse cell (same relative geometry for 480px sim and 224px dataset frames)."""
    side = int(round(SIDE480[cam] * size / 480)); cs = size / 8; cx, cy = (cell % 8 + 0.5) * cs, (cell // 8 + 0.5) * cs
    return int(round(min(max(cx - side / 2, 0), size - side))), int(round(min(max(cy - side / 2, 0), size - side))), side


@torch.no_grad()
def frozen_coarse(model, tok, mod, bs=32):
    W, V = [], []
    for i in range(0, tok.shape[0], bs):
        t = tok[i:i + bs].float().cuda(); m = mod[i:i + bs].long().cuda(); w, x = model.weights(t, m); W.append(w.cpu().half()); V.append(model.Wv(x).cpu().half())
    return torch.cat(W), torch.cat(V)


def candidates(w):  # w [N,192] -> cell [N,4] (front r1,r2, right r1,r2), mass [N,4] (coarse mass in the 3x3 neighbourhood)
    N = w.shape[0]; cell = np.zeros((N, 4), np.int64); mass = np.zeros((N, 4), np.float32); wn = w.float().numpy()
    for n in range(N):
        for cam in (0, 1):
            m = wn[n, cam * 64:(cam + 1) * 64].reshape(8, 8)
            for r, c in enumerate(nms_peaks(m, 2)):
                cell[n, cam * 2 + r] = c; rr, cc = c // 8, c % 8; mass[n, cam * 2 + r] = m[max(0, rr - 1):rr + 2, max(0, cc - 1):cc + 2].sum()
    return cell, mass


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coarse = Bottleneck().cuda(); coarse.load_state_dict(torch.load(COARSE)); coarse.eval()
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); vlm = policy.backbone.model.vlm_with_expert; scale = math.sqrt(960); cfg = VLABenchEnv(task="select_poker")
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    env_pre, _ = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy.config)
    sim = TaskData("select_poker", "cpu"); S = len(sim.sids); B = {}
    for p in sorted(PD.glob("bboxes_*.json")): B.update(json.load(open(p)))
    # ---------------- sim: coarse + candidates ----------------
    w, v = frozen_coarse(coarse, sim.tok.reshape(S * 4, *sim.tok.shape[2:]), sim.mod.reshape(S * 4, -1)); w = w.reshape(S, 4, 192); v = v.reshape(S, 4, 192, 256)
    cell, mass = candidates(w.reshape(S * 4, 192)); cell = cell.reshape(S, 4, 4); mass = mass.reshape(S, 4, 4); valid = sim.valid.numpy()
    # ---------------- dataset: coarse + candidates ----------------
    ds = {}
    for name, f in (("dtrain", "g3_train_dataset_cache.pt"), ("dvald", "g3_vald_cache.pt"), ("dheld", "g3_held_cache.pt")):
        c = torch.load(G3E / f, weights_only=False); n, V = c["tokens"].shape[:2]; dw, dv = frozen_coarse(coarse, c["tokens"].reshape(n * V, *c["tokens"].shape[2:]), c["modality"].reshape(n * V, -1)); dcell, dmass = candidates(dw)
        ds[name] = {"w": dw.reshape(n, V, 192), "v": dv.reshape(n, V, 192, 256), "cell": dcell.reshape(n, V, 4), "mass": dmass.reshape(n, V, 4), "anchors": c["anchors"], "meta": c["meta"]}
        print(name, "coarse+candidates", n, V, flush=True)
    # ---------------- crop list ----------------
    keys = {}

    def kid(k): return keys.setdefault(k, len(keys))

    sim_crop = np.zeros((S, 4, 4), np.int64) - 1
    for s in range(S):
        for k in range(4):
            if valid[s, k]:
                for j in range(4): sim_crop[s, k, j] = kid(("sim", sim.sids[s], j // 2, int(cell[s, k, j])))
    for name, d in ds.items():
        n, V = d["cell"].shape[:2]; d["crop"] = np.zeros((n, V, 4), np.int64)
        for i in range(n):
            for t in range(V):
                for j in range(4): d["crop"][i, t, j] = kid((name, i, j // 2, int(d["cell"][i, t, j])))
    print("unique crops:", len(keys), flush=True)
    # ---------------- encode crops ----------------
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dset = LeRobotDataset(REPO_ID); from_idx = dset.meta.episodes["dataset_from_index"]; cams = [dset.meta.camera_keys[0], dset.meta.camera_keys[1]]
    eps = pd.read_json(D0 / "episodes.json").set_index("episode"); by_img = {}
    for key, idx in keys.items(): by_img.setdefault((key[0], key[1]), []).append((key, idx))
    emb = torch.zeros(len(keys), 960, dtype=torch.float16); embc = torch.zeros(len(keys), 960, dtype=torch.float16); done = 0

    @torch.no_grad()
    def embed(imgs):  # list of uint8 HxWx3 -> [n,960] mean, [n,960] centre-4x4 mean
        t = torch.stack([torch.from_numpy(x).permute(2, 0, 1).float() / 255.0 for x in imgs]).cuda() if len({x.shape for x in imgs}) == 1 else None
        outs = []
        for x in imgs:
            tt = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0; outs.append(resize_with_pad(tt, 512, 512, pad_value=0) * 2.0 - 1.0)
        e = vlm.embed_image(torch.cat(outs)).float() * scale; g = e.reshape(len(imgs), 8, 8, 960); return e.mean(1).cpu(), g[:, 2:6, 2:6].mean((1, 2)).cpu()

    for (dom, img), lst in by_img.items():
        if dom == "sim": rgb = np.load(PD / f"{img}.npz")["rgb"]; size = 480; frames = [rgb[0], rgb[1]]
        else:
            if dom == "dtrain": m = ds[dom]["meta"][img]; fr = int(eps.loc[m["episode"], "t_close"]) - int(m["k"]); ep = m["episode"]
            else: m = ds[dom]["meta"][img]; fr = int(m["image_frame"]); ep = m["episode"]
            item = dset[int(from_idx[int(ep)]) + fr]; frames = [item[c].permute(1, 2, 0).numpy() for c in cams]; frames = [(f * 255).astype(np.uint8) if f.dtype != np.uint8 else f for f in frames]; size = frames[0].shape[0]
        crops = []
        for key, idx in lst:
            x0, y0, side = crop_geom(key[3], key[2], size); crops.append(frames[key[2]][y0:y0 + side, x0:x0 + side])
        em, ec = embed(crops)
        for (key, idx), a, b_ in zip(lst, em, ec): emb[idx] = a.half(); embc[idx] = b_.half()
        done += 1
        if done % 500 == 0: print("images", done, "/", len(by_img), flush=True)
    # ---------------- sanity: re-encode 60 random crops ----------------
    rng = np.random.default_rng(0); sims = []; lst = [(k, i) for k, i in keys.items() if k[0] == "sim"]
    for j in rng.choice(len(lst), 60, replace=False):
        key, idx = lst[j]; rgb = np.load(PD / f"{key[1]}.npz")["rgb"]; x0, y0, side = crop_geom(key[3], key[2], 480); e, _ = embed([rgb[key[2]][y0:y0 + side, x0:x0 + side]])
        sims.append(float(torch.nn.functional.cosine_similarity(e[0], emb[idx].float(), dim=0)))
    print("re-encode cosine (60 crops): min %.6f mean %.6f" % (min(sims), np.mean(sims)), flush=True)
    # ---------------- sim purity diagnostics (GT used for diagnostics only) ----------------
    P = {k: np.zeros((S, 4, 4), np.float32) for k in ("contains_target", "n_cards_inside", "target_frac_of_crop", "distractor_frac_of_crop", "target_bbox_coverage", "center_dist_px", "target_nearest_to_center", "nearest_card_is_target")}
    for s in range(S):
        sid = sim.sids[s]; names = sim.names[s]
        for k in range(4):
            if not valid[s, k]: continue
            for j in range(4):
                cam = j // 2; x0, y0, side = crop_geom(int(cell[s, k, j]), cam, 480); cx, cy = x0 + side / 2, y0 + side / 2; area = side * side; ins, inter, cd = [], {}, {}
                for c in range(len(names)):
                    if not valid[s, c]: continue
                    bb = B[sid]["boxes_xyxy_per_camera"][names[c]][cam]; bcx, bcy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2; cd[c] = math.hypot(bcx - cx, bcy - cy)
                    inter[c] = max(0, min(x0 + side, bb[2]) - max(x0, bb[0])) * max(0, min(y0 + side, bb[3]) - max(y0, bb[1]))
                    if x0 <= bcx <= x0 + side and y0 <= bcy <= y0 + side: ins.append(c)
                bb = B[sid]["boxes_xyxy_per_camera"][names[k]][cam]; barea = (bb[2] - bb[0]) * (bb[3] - bb[1])
                P["contains_target"][s, k, j] = float(k in ins); P["n_cards_inside"][s, k, j] = len(ins); P["target_frac_of_crop"][s, k, j] = inter[k] / area; P["distractor_frac_of_crop"][s, k, j] = sum(v_ for c, v_ in inter.items() if c != k) / area
                P["target_bbox_coverage"][s, k, j] = inter[k] / barea; P["center_dist_px"][s, k, j] = cd[k]; P["nearest_card_is_target"][s, k, j] = float(min(cd, key=cd.get) == k)
                P["target_nearest_to_center"][s, k, j] = float(k in ins and min(ins, key=lambda c: cd[c]) == k)
    # ---------------- text token embeddings (frozen VLM embedding table) ----------------
    recs = {sid_: json.load(open(PD / f"{sid_}.json")) for sid_ in sim.sids}
    tset = {c["instruction"] for r in recs.values() for c in r["candidates"]} | {m["text"] for m in ds["dtrain"]["meta"]}
    for k in ("dvald", "dheld"): tset |= {t for m in ds[k]["meta"] for t in m["texts"]}
    texts = sorted(tset)
    z = np.load(PD / f"{sim.sids[0]}.npz"); raw = {"pixels": {k: z["rgb"][i][None] for i, k in enumerate(("image", "second_image", "wrist_image"))}, "agent_pos": z["state"][None].astype(np.float64)}; toks = []
    for t in texts:
        o = preprocess_observation(copy.deepcopy(raw)); o["task"] = [t]; b = pre(env_pre(o)); ids = b["observation.language.tokens"][0]; m = b["observation.language.attention_mask"][0].bool()
        with torch.no_grad(): toks.append((vlm.embed_language_tokens(ids[m].unsqueeze(0))[0].float() * scale).cpu())
    Tm = max(t.shape[0] for t in toks); tok_emb = torch.zeros(len(texts), Tm, 960, dtype=torch.float16); tok_mask = torch.zeros(len(texts), Tm, dtype=torch.bool)
    for i, t in enumerate(toks): tok_emb[i, :t.shape[0]] = t.half(); tok_mask[i, :t.shape[0]] = True
    tid = {t: i for i, t in enumerate(texts)}
    sim_instr = np.array([[tid[recs[sim.sids[s]]["candidates"][k]["instruction"]] if k < len(recs[sim.sids[s]]["candidates"]) else -1 for k in range(4)] for s in range(S)])
    for name, d in ds.items(): d["instr"] = np.array([[tid[t] for t in (m["texts"] if "texts" in m else [m["text"]])] for m in d["meta"]])
    cache = {"sim": {"sids": sim.sids, "split": list(sim.split), "names": sim.names, "valid": sim.valid, "anchors": sim.anchors if hasattr(sim, "anchors") else sim.anc.cpu(), "cellfrac": sim.cf.cpu(), "held": sim.held.cpu(), "w": w, "v": v,
                    "cell": cell, "mass": mass, "crop": sim_crop, "purity": P, "instr": sim_instr}, **ds, "crops": {"emb": emb, "emb_center": embc, "keys": [None] * len(keys)}, "text": {"texts": texts, "tok_emb": tok_emb, "tok_mask": tok_mask},
             "consts": {"pos_mean": sim.mean.cpu(), "pos_std": sim.std.cpu()}, "crop_side_480": SIDE480}
    for k, i in keys.items(): cache["crops"]["keys"][i] = k
    torch.save(cache, OUT / "g4_cache.pt"); print("saved", {k: (tuple(x.shape) if hasattr(x, "shape") else None) for k, x in (("emb", emb), ("tok_emb", tok_emb))}, "n texts", len(texts), flush=True)


if __name__ == "__main__":
    main()
