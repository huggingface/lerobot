"""[StagePlan-v2.1] (18) language-vs-vision causal test + (19) layer-by-layer language authority, on pilot VAL scenes (same pixels, only text changes).
Conditions per (scene, referent i): A correct; B wrong visible referent j (all j != i); C unrelated instruction (other task family); D shuffled words; E blank.
Usage: uv run python examples/safediff_vla/v21_eval_causal.py --head v2|<dir> --out <json>"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from v21_common import OUT, XYZ_TASKS, load_policy, make_batch, offsets  # noqa: E402
from v21_eval_offline import load_head  # noqa: E402

UNRELATED = {"select_poker": "primitive: Take out the jumpstart from the fridge_open", "select_mahjong": "primitive: Take out the jumpstart from the fridge_open", "select_drink": "primitive: Please pick the poker 9 of diamonds", "select_book": "primitive: Pick up the mahjong of 7 man", "select_fruit": "primitive: Please pick the poker 9 of diamonds", "select_toy": "primitive: Pick up the mahjong of 7 man"}


@torch.no_grad()
def run(pol, pre, head, r, text, aux=True):
    bb = pre(make_batch(r, text)); tok, pad, mod = pol._encode_multimodal_latent(bb); o = head(tok, pad, mod, pol._current_state(bb), return_aux=aux) if aux and hasattr(head, "ref_xyz") else head(tok, pad, mod, pol._current_state(bb))
    o["_tok"] = tok; o["_mod"] = mod; o["g_phys"] = (o["grasp_xyz"] * pol.action_pos_std + pol.action_pos_mean)[0].cpu().numpy().astype(np.float64); return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--head", default="v2"); ap.add_argument("--out", default=None); ap.add_argument("--rng", type=int, default=0); a = ap.parse_args()
    rows = pickle.load(open(OUT / "referent_table.pkl", "rb")); split = json.load(open(OUT / "pilot_scene_split.json")); off = offsets(); rng = np.random.default_rng(a.rng)
    pol, pre = load_policy(); head = load_head(a.head, pol._multimodal_latent_dim()); v21 = hasattr(head, "ref_xyz")
    cond = {c: {} for c in "ABCDE"}; layer = {}
    for r in rows:
        if split[r["scene_id"]] != "val" or not r["scene_ok"]: continue
        C = [c for c in r["candidates"] if c["valid"]]
        if len(C) < 2: continue
        o_ = off.get(r["task"], np.zeros(3)) if r["task"] in XYZ_TASKS else np.zeros(3); A = np.stack([c["anchor"] for c in C]) + o_; tk = r["task"]
        outs = [run(pol, pre, head, r, c["instruction"]) for c in C]
        for i, c in enumerate(C):
            words = c["instruction"].split(); sh = words.copy(); rng.shuffle(sh)
            variants = {"C": UNRELATED[tk], "D": " ".join(sh), "E": ""}
            gi = outs[i]["g_phys"]; d = cond["A"].setdefault(tk, {"err": [], "shift": [], "tcos": [], "nearest_other_visible": [], "to_target_frac": []}); d["err"].append(np.linalg.norm(gi - A[i]))
            for j in range(len(C)):
                if j == i: continue
                g = A[j] - A[i]; s = outs[j]["g_phys"] - gi; db = cond["B"].setdefault(tk, {"err_to_new": [], "cos": [], "real": []}); db["err_to_new"].append(np.linalg.norm(outs[j]["g_phys"] - A[j])); db["cos"].append(float(g @ s / (np.linalg.norm(g) * np.linalg.norm(s) + 1e-9))); db["real"].append(float(s @ g / (g @ g)))
            for k, txt in variants.items():
                o = run(pol, pre, head, r, txt); dd = cond[k].setdefault(tk, {"shift_cm": [], "err_to_orig_target_cm": [], "nearest_is_visible_cand": [], "nearest_cand_is_orig": []}); gk = o["g_phys"]
                D = np.linalg.norm(A - gk, axis=1); dd["shift_cm"].append(np.linalg.norm(gk - gi) * 100); dd["err_to_orig_target_cm"].append(np.linalg.norm(gk - A[i]) * 100); dd["nearest_is_visible_cand"].append(float(D.min() < 0.05)); dd["nearest_cand_is_orig"].append(float(D.argmin() == i))
        if v21 and len(C) >= 2:  # (19) layer audit on the first ordered pair
            oa, ob = outs[0], outs[1]; rel = lambda x, y: float((x - y).float().norm() / (x.float().norm() + 1e-9)); L = layer.setdefault(tk, {k: [] for k in ("backbone_img_tokens", "backbone_txt_tokens_rel", "lang_vec", "spatial_scores", "alpha_TV", "ref_coords_shift_tokens", "pooled_feature", "grasp_hidden", "ref_xyz_cm", "grasp_xyz_cm")})
            im = (oa["_mod"][0] == 0); L["backbone_img_tokens"].append(rel(oa["_tok"][0][im], ob["_tok"][0][im])); L["lang_vec"].append(rel(oa["lang"], ob["lang"])); L["spatial_scores"].append(rel(oa["logits"], ob["logits"]))
            L["alpha_TV"].append(float(0.5 * (oa["alpha"].float() - ob["alpha"].float()).abs().sum(-1).mean())); L["ref_coords_shift_tokens"].append(float((oa["ref_coords"] - ob["ref_coords"]).float().norm(dim=-1)[:, :2].mean()))
            L["pooled_feature"].append(rel(oa["pooled"], ob["pooled"])); L["grasp_hidden"].append(rel(oa["h"], ob["h"])); L["ref_xyz_cm"].append(float((oa["ref_xyz"] - ob["ref_xyz"]).norm() * pol.action_pos_std.mean().item() * 100)); L["grasp_xyz_cm"].append(float(np.linalg.norm(oa["g_phys"] - ob["g_phys"]) * 100))
    res = {"conditions": {c: {t: {k: float(np.mean(v)) for k, v in d.items()} for t, d in cond[c].items()} for c in cond}, "layer_audit": {t: {k: float(np.mean(v)) for k, v in d.items()} for t, d in layer.items()}}
    for c, d in res["conditions"].items():
        print(f"== condition {c}")
        for t, m in d.items(): print(f"  {t:15s}", {k: round(v, 3) for k, v in m.items()})
    if layer:
        print("== layer-by-layer instruction sensitivity (same scene, referent 0 -> 1)")
        for t, m in res["layer_audit"].items(): print(f"  {t:15s}", {k: round(v, 3) for k, v in m.items()})
    if a.out: json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__": main()
