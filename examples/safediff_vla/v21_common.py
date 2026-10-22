"""[StagePlan-v2.1] shared: scene-level split, frozen-backbone token cache for pilot (scene, instruction) samples, counterfactual metrics."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.stageplan_v2 import SafeDiffVLAStagePlanV2Policy  # noqa: E402

CKPT = "outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/020000/pretrained_model"
OUT = Path("outputs/stageplan_v21"); SPLIT_SEED = 20210; VAL_FRAC = 0.2
XYZ_TASKS = ("select_poker", "select_mahjong", "select_book", "select_fruit", "select_toy")  # expert offset reliable (p90 dev <= 3 cm); drink excluded from L_grasp
CAMS = ["image", "second_image", "wrist_image"]


def load_policy(dev="cuda"):
    pol = SafeDiffVLAStagePlanV2Policy.from_pretrained(CKPT).to(dev).eval()
    for p in pol.parameters(): p.requires_grad_(False)
    pre, _ = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=CKPT, preprocessor_overrides={"device_processor": {"device": dev}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    return pol, pre


def scene_split(rows):
    """Per task: seeded permutation of scenes, last 20% = val. ALL counterfactual instructions of a scene share the split."""
    split = {}
    for task in sorted({r["task"] for r in rows}):
        ids = sorted(r["scene_id"] for r in rows if r["task"] == task); rng = np.random.default_rng(SPLIT_SEED); perm = rng.permutation(len(ids)); nv = int(round(VAL_FRAC * len(ids)))
        for j, i in enumerate(perm): split[ids[i]] = "val" if j < nv else "train"
    return split


def offsets():
    o = json.load(open(OUT / "expert_offsets.json")); return {f"select_{k}": np.array(v["median_offset_m"]) for k, v in o.items()}


def make_batch(r, text):
    b = {f"observation.images.{k}": torch.from_numpy(r["imgs224"][i]).permute(2, 0, 1).float().div(255)[None] for i, k in enumerate(CAMS)}
    b["observation.state"] = torch.from_numpy(r["state"]).float()[None]; b["task"] = [text]; return b


@torch.no_grad()
def build_cache(pol, pre, rows, path):
    """tokens fp16 [N,205ish,960] padded to 256, pad mask, modality ids, normalised state; one entry per (scene, candidate instruction)."""
    ent = []
    for r in rows:
        if not r["scene_ok"]: continue
        for ci, c in enumerate(r["candidates"]):
            bb = pre(make_batch(r, c["instruction"])); tok, pad, mod = pol._encode_multimodal_latent(bb); n = tok.shape[1]
            T = torch.zeros(1, 256, tok.shape[2], dtype=torch.float16); T[:, :n] = tok.half().cpu(); P = torch.zeros(1, 256, dtype=torch.bool); P[:, :n] = pad.cpu().bool(); M = torch.full((1, 256), 2, dtype=torch.long); M[:, :n] = mod.cpu()
            ent.append({"scene_id": r["scene_id"], "task": r["task"], "ci": ci, "name": c["name"], "tok": T[0], "pad": P[0], "mod": M[0], "state": pol._current_state(bb)[0].cpu(), "anchor": c["anchor"], "valid": c["valid"]})
    torch.save(ent, path); return ent


def cf_metrics(pred_by_key, rows, split, which="val", off=None, key="grasp"):
    """pred_by_key[(scene_id, ci)] = predicted xyz (phys, robot frame). Same-scene ordered pairs (i->j): swap instruction i -> j."""
    off = off or offsets(); per = {}
    for r in rows:
        if split[r["scene_id"]] != which or not r["scene_ok"]: continue
        C = [c for c in r["candidates"] if c["valid"]]; ids = [i for i, c in enumerate(r["candidates"]) if c["valid"]]
        if len(C) < 2: continue
        o = off.get(r["task"], np.zeros(3)) if r["task"] in XYZ_TASKS else np.zeros(3)
        if key == "ref": o = np.zeros(3)
        A = np.stack([c["anchor"] for c in C]) + o; P = np.stack([pred_by_key[(r["scene_id"], i)] for i in ids]); d = per.setdefault(r["task"], {"acc": [], "chance": [], "cos": [], "real": [], "gt": [], "pr": [], "err": []})
        D = np.linalg.norm(P[:, None] - A[None], axis=-1)
        for a in range(len(C)):
            d["acc"].append(float(D[a].argmin() == a)); d["chance"].append(1 / len(C)); d["err"].append(float(D[a, a]))
            for b in range(len(C)):
                if a == b: continue
                g, p = A[b] - A[a], P[b] - P[a]; ng = np.linalg.norm(g)
                if ng < 0.03: continue
                d["cos"].append(float(g @ p / (ng * np.linalg.norm(p) + 1e-9))); d["real"].append(float(p @ g / ng ** 2)); d["gt"].append(ng); d["pr"].append(float(np.linalg.norm(p)))
    res = {}
    for t, d in per.items():
        err = np.array(d["err"]) * 100
        res[t] = {"n_pairs": len(d["cos"]), "sel_acc": float(np.mean(d["acc"])), "chance": float(np.mean(d["chance"])), "gt_disp_cm": float(np.mean(d["gt"]) * 100), "pred_disp_cm": float(np.mean(d["pr"]) * 100), "cos": float(np.mean(d["cos"])),
                  "realized": float(np.mean(d["real"])), "cos>0.5": float(np.mean(np.array(d["cos"]) > 0.5)), "real>0.5": float(np.mean(np.array(d["real"]) > 0.5)), "own_err_med_cm": float(np.median(err)), "own<3cm": float(np.mean(err < 3)), "own<5cm": float(np.mean(err < 5))}
    return res
