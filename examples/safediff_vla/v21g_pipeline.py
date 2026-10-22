#!/usr/bin/env python
"""[v2.1 anchor phase] Deployable offline path: RGB -> (frozen front proposals g64 thr 0.10 + right-camera proposals) -> calibrated two-view triangulation (3D reference) -> frozen HR referent scorer
-> shared local-anchor offset -> grasp_xyz (robot frame). NO simulator xyz at inference. Proposal heads and scorer are frozen (the right head exists only for triangulation).
GT grasp = keypoint anchor + expert (EE - keypoint) median offset (label construction only). Methods: A = triangulated centre only; B = centre + ONE global median offset (shared, from train);
C = shared learned anchor head (selected crops of both views + triangulated 3D -> offset), kept ONLY IF dev median error beats B by >=10% on BOTH tasks (pre-declared). Diagnostics (not methods): per-task / per-identity median offset, GT-2D-centre triangulation.
Anchor head: seed 1000, 3000 steps, batch 64, AdamW 3e-4/1e-2, FINAL step. Test block C is scored once after the keep/drop decision.
Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/v21g_pipeline.py"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).parent))
import v21hr_candidate as H  # noqa: E402
import v21p_proposals as P0  # noqa: E402
import v21q_prop_iter as Q  # noqa: E402
import v21x_hr as XH  # noqa: E402
import v21g_right_prop as R  # noqa: E402
from v21_common import OUT as V21, load_policy, offsets  # noqa: E402
from v21g_geometry import CAMS, project, triangulate  # noqa: E402

X = V21 / "x"; TASKS = H.TASKS; THR = 0.10; GAP_MAX = 0.03; MATCH = Q.MATCH_PX; FAIL_CM = 3.0


class AnchorHead(nn.Module):
    D = 128
    def __init__(self):
        super().__init__(); D = self.D; self.vis = nn.Sequential(nn.LayerNorm(960), nn.Linear(960, D)); self.pos = nn.Parameter(torch.randn(2, 64, D) * .02); self.q = nn.Parameter(torch.randn(1, 1, D) * .02); self.attn = nn.MultiheadAttention(D, 4, batch_first=True, dropout=0.1)
        self.geo = nn.Sequential(nn.Linear(3, 64), nn.GELU()); self.mlp = nn.Sequential(nn.Linear(2 * D + 64, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 3))
    def forward(self, tf, tr, ref):  # tf,tr [B,64,960]; ref [B,3] metres (robot frame)
        outs = []
        for i, t in enumerate((tf, tr)): v = self.vis(t) + self.pos[i]; o, _ = self.attn(self.q.expand(len(t), -1, -1), v, v, need_weights=False); outs.append(o.squeeze(1))
        return self.mlp(torch.cat(outs + [self.geo((ref - torch.tensor([0, 0.45, 0.05], device=ref.device)) / 0.2)], -1))  # offset in cm


def gt_grasp(r, c, off): return c["anchor"] + off[r["task"]]


def props(fm, rm, ffe, rfe, rows):
    of, orr = Q.infer(fm, ffe, rows), Q.infer(rm, rfe, rows); return {r["scene_id"]: (Q.decode(of[r["scene_id"]], 64, THR), Q.decode(orr[r["scene_id"]], 64, THR)) for r in rows}


def associate(Pf, Pr):
    """returns {front_idx: (ref3d_robot, right_idx, gap)} via Hungarian on two-ray gap; pairs with gap > GAP_MAX or outside loose workspace bounds rejected."""
    if not Pf or not Pr: return {}
    C = np.full((len(Pf), len(Pr)), 1e3); pts = {}
    for i, f in enumerate(Pf):
        for j, g in enumerate(Pr):
            p, gap = triangulate(f[:2], g[:2])
            if p is not None: C[i, j] = gap; pts[(i, j)] = p
    a, b = linear_sum_assignment(C); out = {}
    for i, j in zip(a, b):
        if C[i, j] <= GAP_MAX: out[i] = (pts[(i, j)] - np.array([0, -0.4, 0.78]), j, C[i, j])
    return out


def match_gt(Pp, G):
    """greedy one-to-one proposal->GT (within MATCH px) -> {prop_idx: gt_idx}"""
    pairs = sorted([(np.hypot(p[0] - g[0], p[1] - g[1]), i, j) for i, p in enumerate(Pp) for j, g in enumerate(G)]); used_p, used_g, m = set(), set(), {}
    for d, i, j in pairs:
        if d <= MATCH and i not in used_p and j not in used_g: m[i] = j; used_p.add(i); used_g.add(j)
    return m


class Ctx:
    def __init__(self, pol, pre, sc, fm, rm): self.pol, self.pre, self.sc, self.fm, self.rm = pol, pre, sc, fm, rm; self.ve = pol.backbone.model.vlm_with_expert
    @torch.no_grad()
    def crop_tok(self, r, cam, u, v):
        u, v = int(round(min(max(u, 0), 479))), int(round(min(max(v, 0), 479))); h = H.CROP_PX // 2; pad = np.pad(r["imgs480"][cam], ((h, h), (h, h), (0, 0)), mode="edge"); crop = pad[v: v + H.CROP_PX, u: u + H.CROP_PX]
        return self.ve.embed_image(torch.from_numpy(cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).float().div(255)[None].cuda() * 2 - 1)[0].float()


def scene_struct(ctx, r, Pf, Pr):
    gf = R.centers_cam(r, 0); gr = R.centers_cam(r, 1); ids = [i for i, c in enumerate(r["candidates"]) if c["valid"]]; assoc = associate(Pf, Pr); mf, mr = match_gt(Pf, gf), match_gt(Pr, gr)
    return {"gf": gf, "gr": gr, "ids": ids, "assoc": assoc, "mf": mf, "mr": mr, "ftok": {}, "rtok": {}, "Pf": Pf, "Pr": Pr}


def tok_f(ctx, r, S, i):
    if i not in S["ftok"]: S["ftok"][i] = ctx.crop_tok(r, 0, *S["Pf"][i][:2])
    return S["ftok"][i]


def tok_r(ctx, r, S, j):
    if j not in S["rtok"]: S["rtok"][j] = ctx.crop_tok(r, 1, *S["Pr"][j][:2])
    return S["rtok"][j]


def train_dataset(ctx, rows, P, off):
    D = []
    for r in rows:
        Pf, Pr = P[r["scene_id"]]; S = scene_struct(ctx, r, Pf, Pr); inv = {g: i for i, g in S["mf"].items()}
        for k, ci in enumerate(S["ids"]):
            if k in inv and inv[k] in S["assoc"]:
                i = inv[k]; ref, j, _ = S["assoc"][i]; D.append({"task": r["task"], "name": r["candidates"][ci]["name"], "tf": tok_f(ctx, r, S, i).half().cpu(), "tr": tok_r(ctx, r, S, j).half().cpu(), "ref": ref, "res": (gt_grasp(r, r["candidates"][ci], off) - ref) * 100})
    return D


def fit_anchor(D):
    torch.manual_seed(H.SEED); rng = np.random.default_rng(H.SEED); m = AnchorHead().cuda(); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-2); TF = torch.stack([d["tf"] for d in D]); TR = torch.stack([d["tr"] for d in D]); REF = torch.tensor(np.stack([d["ref"] for d in D]), dtype=torch.float32); Y = torch.tensor(np.stack([d["res"] for d in D]), dtype=torch.float32)
    for step in range(1, 3001):
        m.train(); idx = torch.from_numpy(rng.choice(len(D), 64, replace=False)); loss = F.smooth_l1_loss(m(TF[idx].float().cuda(), TR[idx].float().cuda(), REF[idx].cuda()), Y[idx].cuda()); opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0: print("anchor step", step, float(loss), flush=True)
    return m.eval()


def stats(e):
    e = np.asarray([x for x in e], float); f = np.isfinite(e)
    return {"n": int(len(e)), "mean_cm": float(np.mean(e[f])) if f.any() else None, "median_cm": float(np.median(e[f])) if f.any() else None, "p90_cm": float(np.percentile(e[f], 90)) if f.any() else None, **{f"<{t}cm": float(np.mean(e < t)) for t in (1, 2, 3, 5)}}


@torch.no_grad()
def evaluate_set(ctx, rows, P, off, offB, headC, tt, train_counts):
    res = {t: {"gt_candidate": {k: [] for k in ("A", "B", "C", "B_taskmedian_diag", "oracle_gt2d_B_diag")}, "full": {"B": [], "C": []}, "decomp": {"B": {}, "C": {}}, "swap": {"B": [], "C": []}, "ref_err_cm": [], "n_cand": 0, "n_cand_with_3d": 0, "referent": [], "per_identity": {}, "n_instr": 0} for t in TASKS}
    tmed = {t: np.median(np.stack([d["res"] for d in tt if d["task"] == t]), 0) / 100 for t in TASKS}
    Tt = XH.text_tokens(ctx.pol, ctx.pre, rows); ve = ctx.ve
    def offs(S, r, i):
        ref, j, _ = S["assoc"][i]; oc = headC(tok_f(ctx, r, S, i)[None], tok_r(ctx, r, S, j)[None], torch.tensor(ref, dtype=torch.float32, device="cuda")[None])[0].cpu().numpy() / 100; return ref, offB, oc
    for r in rows:
        t = r["task"]; R_ = res[t]; Pf, Pr = P[r["scene_id"]]; S = scene_struct(ctx, r, Pf, Pr); inv = {g: i for i, g in S["mf"].items()}; ids = S["ids"]; GT = {k: gt_grasp(r, r["candidates"][ci], off) for k, ci in enumerate(ids)}
        for k, ci in enumerate(ids):  # (7/8) GT-candidate anchor quality
            R_["n_cand"] += 1; c = r["candidates"][ci]; w = c["obj_center"] + r["base"]
            pf, pr = project(CAMS[0], w), project(CAMS[1], w); p, _ = triangulate(pf[:2], pr[:2]); R_["gt_candidate"]["oracle_gt2d_B_diag"].append(np.linalg.norm((p - r["base"] + offB) - GT[k]) * 100)
            if k in inv and inv[k] in S["assoc"]:
                R_["n_cand_with_3d"] += 1; ref, ob, oc = offs(S, r, inv[k]); R_["ref_err_cm"].append(np.linalg.norm(ref - c["obj_center"]) * 100)
                for name, v in (("A", ref), ("B", ref + ob), ("C", ref + oc), ("B_taskmedian_diag", ref + tmed[t])): R_["gt_candidate"][name].append(np.linalg.norm(v - GT[k]) * 100)
            else:
                for name in ("A", "B", "C", "B_taskmedian_diag"): R_["gt_candidate"][name].append(np.inf)
        # full pipeline per instruction
        outs = {"B": {}, "C": {}}
        for k, ci in enumerate(ids):
            R_["n_instr"] += 1; txt = Tt[(r["scene_id"], ci)][None].cuda(); tm = torch.ones(1, txt.shape[1], dtype=torch.bool, device="cuda")
            covered = k in inv; name = r["candidates"][ci]["name"]
            if not Pf:
                R_["referent"].append(0.0); sel = None
            else:
                lg = ctx.sc(txt, tm, torch.stack([tok_f(ctx, r, S, i) for i in range(len(Pf))])[None])[0]; sel = int(lg.argmax()); R_["referent"].append(float(S["mf"].get(sel, -1) == k))
            pi = R_["per_identity"].setdefault(name, {"n": 0, "ref_ok": 0, "err_B": [], "err_gtcand_B": []}); pi["n"] += 1; pi["ref_ok"] += R_["referent"][-1]
            if covered and inv[k] in S["assoc"]: pi["err_gtcand_B"].append(R_["gt_candidate"]["B"][-1 - (len(ids) - 1 - k)])
            for mth in ("B", "C"):
                if sel is None or sel not in S["assoc"]: pred = None
                else: ref, ob, oc = offs(S, r, sel); pred = ref + (ob if mth == "B" else oc)
                outs[mth][k] = pred; err = np.linalg.norm(pred - GT[k]) * 100 if pred is not None else np.inf; R_["full"][mth].append(err)
                if mth == "B": pi["err_B"].append(err)
                if not covered: cat = "A_proposal_miss"
                elif S["mf"].get(sel, -1) != k: cat = "B_wrong_referent"
                elif pred is None or np.linalg.norm(S["assoc"][sel][0] - r["candidates"][ci]["obj_center"]) > 0.02: cat = "C_bad_3D_localization"
                elif err >= FAIL_CM: cat = "D_bad_local_offset"
                else: cat = "success"
                R_["decomp"][mth][cat] = R_["decomp"][mth].get(cat, 0) + 1
        for mth in ("B", "C"):  # (13) same-scene swaps through the FULL pipeline
            for a in range(len(ids)):
                for b in range(len(ids)):
                    if a == b: continue
                    g = GT[b] - GT[a]; ng = np.linalg.norm(g)
                    if ng < 0.03: continue
                    pa, pb = outs[mth][a], outs[mth][b]; d = (pb - pa) if (pa is not None and pb is not None) else np.zeros(3); nd = np.linalg.norm(d)
                    R_["swap"][mth].append({"cos": float(g @ d / (ng * nd)) if nd > 1e-9 else 0.0, "real": float(d @ g / ng ** 2), "gt_cm": ng * 100, "pred_cm": nd * 100, "errA": np.linalg.norm(pa - GT[a]) * 100 if pa is not None else np.inf, "errB": np.linalg.norm(pb - GT[b]) * 100 if pb is not None else np.inf})
    out = {}
    for t, d in res.items():
        sw = {m: ({"pairs": len(v), "cos": float(np.mean([x["cos"] for x in v])), "realized": float(np.mean([x["real"] for x in v])), "gt_disp_cm": float(np.mean([x["gt_cm"] for x in v])), "pred_disp_cm": float(np.mean([x["pred_cm"] for x in v])), "errA_median_cm": float(np.median([x["errA"] for x in v])), "errB_median_cm": float(np.median([x["errB"] for x in v])), "frac_both_<3cm": float(np.mean([(x["errA"] < 3) and (x["errB"] < 3) for x in v]))} if v else {}) for m, v in d["swap"].items()}
        out[t] = {"n_candidates": d["n_cand"], "frac_candidates_with_3d_ref": d["n_cand_with_3d"] / d["n_cand"], "ref_3d_err_cm": stats(d["ref_err_cm"]), "gt_candidate_anchor": {k: stats(v) for k, v in d["gt_candidate"].items()}, "referent_acc": float(np.mean(d["referent"])), "n_instr": d["n_instr"],
                  "full_pipeline_anchor": {m: stats(v) for m, v in d["full"].items()}, "error_decomposition_counts": d["decomp"], "same_scene_swap_full_pipeline": sw, "per_identity": {k: {"n": v["n"], "referent_acc": v["ref_ok"] / v["n"], "full_B_median_cm": float(np.median(v["err_B"])), "full_B_lt3": float(np.mean(np.array(v["err_B"]) < 3)), "train_count": train_counts.get(k, 0)} for k, v in d["per_identity"].items()}}
    return out


def main():
    pol, pre = load_policy(); xr = pickle.load(open(X / "referent_table_x.pkl", "rb")); split = json.load(open(X / "split_x.json")); A_, B_, C_ = Q.build_rows("A"), Q.build_rows("B"), Q.build_rows("C"); off = offsets()
    tr = [r for r in xr if split[r["scene_id"]] == "train" and r["scene_ok"]] + [r for r in A_ if r["scene_ok"]]; sets = {"dev": [r for r in B_ if r["scene_ok"]], "test": [r for r in C_ if r["scene_ok"]], "expansion_val_secondary": [r for r in xr if split[r["scene_id"]] == "val" and r["scene_ok"]]}
    fm = Q.Head(64).cuda(); fm.load_state_dict(torch.load(X / "proposal_head_v21q_g64.pt")); fm.eval(); rm = Q.Head(64).cuda(); rm.load_state_dict(torch.load(X / "proposal_head_right_g64.pt")); rm.eval()
    sc = H.Scorer("B").cuda(); sc.load_state_dict(torch.load(X / "scorer_expanded.pt")); sc.eval(); ctx = Ctx(pol, pre, sc, fm, rm); allr = tr + sum(sets.values(), [])
    ffe, rfe = P0.patch_feats(pol, allr), R.feats_cam(pol, allr, 1); P = props(fm, rm, ffe, rfe, allr)
    D = train_dataset(ctx, tr, P, off); print("anchor train samples", len(D), {t: sum(d["task"] == t for d in D) for t in TASKS}, flush=True)
    offB = np.median(np.stack([d["res"] for d in D]), 0) / 100; print("global median offset B (cm)", (offB * 100).round(3), flush=True); headC = fit_anchor(D); torch.save(headC.state_dict(), X / "anchor_head.pt")
    counts = {}
    for r in tr:
        for c in r["candidates"]:
            if c["valid"]: counts[c["name"]] = counts.get(c["name"], 0) + 1
    res = {"frozen": {"front_proposal_head": "outputs/stageplan_v21/x/proposal_head_v21q_g64.pt", "scorer": "outputs/stageplan_v21/x/scorer_expanded.pt", "right_head(new, triangulation only)": "outputs/stageplan_v21/x/proposal_head_right_g64.pt", "threshold": THR, "global_median_offset_cm": (offB * 100).tolist(), "anchor_train_samples": len(D)}}
    res["dev"] = evaluate_set(ctx, sets["dev"], P, off, offB, headC, D, counts); dv = res["dev"]; keepC = all(dv[t]["gt_candidate_anchor"]["C"]["median_cm"] <= 0.9 * dv[t]["gt_candidate_anchor"]["B"]["median_cm"] for t in TASKS)
    res["keep_learned_C"] = bool(keepC); print("DEV", {t: {m: round(dv[t]["gt_candidate_anchor"][m]["median_cm"], 3) for m in ("A", "B", "C")} for t in TASKS}, "keep C:", keepC, flush=True)
    for k in ("test", "expansion_val_secondary"): res[k] = evaluate_set(ctx, sets[k], P, off, offB, headC, D, counts)
    json.dump(res, open(X / "anchor_pipeline_results.json", "w"), indent=1, default=float)
    for k in ("dev", "test", "expansion_val_secondary"):
        for t in TASKS:
            d = res[k][t]; print(k, t, "| 3D-ref frac", round(d["frac_candidates_with_3d_ref"], 3), "ref err med", round(d["ref_3d_err_cm"]["median_cm"], 3), "| GTcand", {m: (round(v["median_cm"], 2), round(v["<3cm"], 3)) for m, v in d["gt_candidate_anchor"].items()}, "| referent", round(d["referent_acc"], 3), "| full", {m: (round(v["median_cm"], 2), round(v["<3cm"], 3)) for m, v in d["full_pipeline_anchor"].items()}, "| decomp B", d["error_decomposition_counts"]["B"], "| swap B", {a: round(b, 3) for a, b in d["same_scene_swap_full_pipeline"]["B"].items()}, flush=True)

if __name__ == "__main__": main()
