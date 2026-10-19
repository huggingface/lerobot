#!/usr/bin/env python
"""G4c cache on top of the G4b candidates (same OOF proposals, same 144px front crops; nothing about candidate generation changes).
Adds: (1) the 64 frozen crop tokens (vision tower + connector, NOT mean pooled) per crop -> g4c_crop_tokens.pt [n_crops,64,960] fp16;
(2) proj[J,64,192]: bilinear weights that scatter each crop-token centre back onto the 8x8 coarse cells of the FRONT camera and (via front ray -> table plane -> right camera) of the RIGHT camera;
(3) sim only: tokmask[k,J,card,64]: area fraction of every card's GT bbox (largest segmentation component) inside each crop token cell -- TRAINING supervision / diagnostics only."""
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
from g4b_geometry import CAL, proj as cam_proj  # noqa: E402
from lerobot.policies.common.vla_utils import resize_with_pad  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

PD = D / "select_poker"; OUT = Path("outputs/eval/safediff_vla_g4b_offline"); OLD = Path("outputs/eval/safediff_vla_g4_offline"); OUTC = Path("outputs/eval/safediff_vla_g4c_offline")


def bilinear(u, v):  # pixel (480 coords) -> dict cell->weight on an 8x8 grid of 60px cells (cell centres at (i+.5)*60), clamped
    cu, cv = np.clip(u / 60 - .5, 0, 7), np.clip(v / 60 - .5, 0, 7); i0, j0 = int(np.floor(cu)), int(np.floor(cv)); i1, j1 = min(i0 + 1, 7), min(j0 + 1, 7); fu, fv = cu - i0, cv - j0
    return [(j0 * 8 + i0, (1 - fu) * (1 - fv)), (j0 * 8 + i1, fu * (1 - fv)), (j1 * 8 + i0, (1 - fu) * fv), (j1 * 8 + i1, fu * fv)]


def proj_matrix(box480):
    x0, y0, sd = box480; M = np.zeros((64, 192), np.float32)
    for t in range(64):
        r, c = divmod(t, 8); px, py = x0 + (c + .5) * sd / 8, y0 + (r + .5) * sd / 8
        for cell, w in bilinear(px, py): M[t, cell] += w
        ru, rv = cam_proj(CAL[1], P.to_plane(0, px, py))
        for cell, w in bilinear(ru, rv): M[t, 64 + cell] += w
    return M


def tok_area(bb, box480):
    x0, y0, sd = box480; ts = sd / 8; out = np.zeros(64, np.float32)
    if bb is None: return out
    for t in range(64):
        r, c = divmod(t, 8); a0, b0 = x0 + c * ts, y0 + r * ts; out[t] = max(0, min(a0 + ts, bb[2]) - max(a0, bb[0])) * max(0, min(b0 + ts, bb[3]) - max(b0, bb[1])) / (ts * ts)
    return out


def main():
    OUTC.mkdir(parents=True, exist_ok=True); old = torch.load(OLD / "g4_cache.pt", weights_only=False); n = torch.load(OUT / "g4b_cache.pt", weights_only=False); s = old["sim"]; S = len(s["sids"]); valid = s["valid"].numpy(); B = {}
    for p in sorted(PD.glob("bboxes_*.json")): B.update(json.load(open(p)))
    out = {}; sb, sv = n["sim"]["box"], n["sim"]["valid"]; proj = np.zeros((S, 4, P.J, 64, 192), np.float16); tm = np.zeros((S, 4, P.J, 4, 64), np.float16)
    for si in range(S):
        sid = s["sids"][si]; names = s["names"][si]
        for k in range(4):
            if not valid[si, k]: continue
            for j in range(P.J):
                if not sv[si, k, j]: continue
                b = tuple(int(v) for v in sb[si, k, j]); proj[si, k, j] = proj_matrix(b)
                for c in range(len(names)):
                    if valid[si, c]: tm[si, k, j, c] = tok_area(B[sid]["boxes_xyxy_per_camera"][names[c]][0], b)
    out["sim"] = {"proj": torch.from_numpy(proj), "tokmask": torch.from_numpy(tm)}; print("sim proj/tokmask done", flush=True)
    for name in ("dtrain", "dvald", "dheld"):
        d = n[name]; nn_, V = d["box"].shape[:2]; pr = np.zeros((nn_, V, P.J, 64, 192), np.float16); f = 480 / 224
        for i in range(nn_):
            for t in range(V):
                for j in range(P.J):
                    if d["valid"][i, t, j]: x0, y0, sd = d["box"][i, t, j]; pr[i, t, j] = proj_matrix((x0 * f, y0 * f, sd * f))
        out[name] = {"proj": torch.from_numpy(pr)}; print(name, "proj done", flush=True)
    torch.save(out, OUTC / "g4c_proj_tokmask.pt")
    # ---- crop tokens ----
    keys = n["crops"]["keys"]; from lerobot.datasets.lerobot_dataset import LeRobotDataset
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); vlm = policy.backbone.model.vlm_with_expert; scale = math.sqrt(960)
    dset = LeRobotDataset(REPO_ID); from_idx = dset.meta.episodes["dataset_from_index"]; cam0 = dset.meta.camera_keys[0]; eps = pd.read_json(D0 / "episodes.json").set_index("episode"); by_img = {}
    for idx, key in enumerate(keys): by_img.setdefault((key[0], key[1]), []).append((key, idx))
    toks = torch.zeros(len(keys), 64, 960, dtype=torch.float16)

    @torch.no_grad()
    def embed(imgs):
        x = torch.cat([resize_with_pad(torch.from_numpy(i).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0, 512, 512, pad_value=0) * 2.0 - 1.0 for i in imgs]); return (vlm.embed_image(x).float() * scale).cpu()

    for n_, ((dom, img), lst) in enumerate(by_img.items()):
        if dom == "sim": fr = np.load(PD / f"{img}.npz")["rgb"][0]
        else:
            m = old[dom]["meta"][img]; f_ = int(eps.loc[m["episode"], "t_close"]) - int(m["k"]) if dom == "dtrain" else int(m["image_frame"]); item = dset[int(from_idx[int(m["episode"])]) + f_]; fr = item[cam0].permute(1, 2, 0).numpy(); fr = (fr * 255).astype(np.uint8) if fr.dtype != np.uint8 else fr
        e = embed([fr[k[3]:k[3] + k[4], k[2]:k[2] + k[4]] for k, _ in lst])
        for (k, idx), a in zip(lst, e): toks[idx] = a.half()
        if (n_ + 1) % 500 == 0: print("images", n_ + 1, "/", len(by_img), flush=True)
    # consistency with the G4b mean embedding (same encoder, same crops)
    mean_chk = (toks[:200].float().mean(1) - n["crops"]["emb"][:200].float()).abs().max().item(); print("max |mean(tokens) - G4b emb| over 200 crops:", mean_chk)
    torch.save(toks, OUTC / "g4c_crop_tokens.pt"); print("saved", tuple(toks.shape))


if __name__ == "__main__":
    main()
