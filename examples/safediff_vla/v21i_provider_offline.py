#!/usr/bin/env python
"""[v2.1 integration gate 8-10] OFFLINE composed-provider test through the REAL provider (examples/safediff_vla/v21_provider.GroundedProvider) on held-out SOURCE scenes (block C, seeds 100440-100499; live env.reset(seed), the
policy's own env/pre-processing, no env.step). For every scene x every candidate instruction: provider -> StageVars. Compared with the frozen StagePlan-v2 grasp (fallback head output), the GT grasp
(keypoint + expert EE offset), and the earlier offline anchor numbers. Executor DRY-RUN: the unchanged `build_pregrasp_chunk` is built from the reset EE pose + the predicted StageVars; trajectory overlays saved.
Privileged simulator reads (keypoints, candidate lists) are used ONLY to score / draw."""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP  # noqa: E402
import v21q_prop_iter as Q  # noqa: E402
from v21_common import CKPT, offsets  # noqa: E402
from v21_provider import GroundedProvider  # noqa: E402
from v21g_pipeline import MATCH  # noqa: E402
import v21p_proposals as P0  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import robot_base  # noqa: E402
from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.stageplan_v2 import SafeDiffVLAStagePlanV2Policy, StageVars, build_pregrasp_chunk  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

X = Path("outputs/stageplan_v21/x"); CA = {t: np.array(v["C-A"]["median_xyz_cm"]) / 100 for t, v in json.load(open(Path("outputs/stageplan_v21/x/label_frame_sim.json"))).items()}; TASKS = ["select_poker", "select_mahjong"]


def st(e):
    e = np.asarray(e, float); f = np.isfinite(e); return {"n": len(e), "median_cm": float(np.median(e[f])), "mean_cm": float(np.mean(e[f])), "p90_cm": float(np.percentile(e[f], 90)), **{f"<{t}cm": float(np.mean(e < t)) for t in (1, 2, 3, 5)}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("--plots", type=int, default=12); a = ap.parse_args(); off = offsets(); C = Q.build_rows("C"); out = {}
    pol = SafeDiffVLAStagePlanV2Policy.from_pretrained(CKPT).to("cuda").eval(); pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=CKPT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    stash = {}; prov = GroundedProvider(pol, lambda: stash["raw"]); (X / "traj").mkdir(exist_ok=True)
    for task in TASKS:
        env_cfg = VLABenchEnv(task=task); env = make_env(env_cfg, n_envs=1, use_async_envs=False)[task][0]; env_pre, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=pol.config); R = {"errC": [], "errC_v2": [], "err": [], "err_v2": [], "src": {}, "cov": [], "ref": [], "decomp": {}, "swap": [], "rows": []}; nplot = 0
        for r in [r for r in C if r["task"] == task and r["scene_ok"]][: a.n]:
            obs, _ = env.reset(seed=[r["seed"]]); e = env.unwrapped.envs[0]; t = e._env.task; ph = e._env.physics; base = robot_base(e); kp = np.asarray(t.entities[t.target_entity].get_grasped_keypoints(ph)[0], float) - base
            tgt = [c for c in r["candidates"] if c["is_target"]][0]; assert np.allclose(kp, tgt["anchor"], atol=1e-6), "scene mismatch vs table"
            stash["raw"] = {k: np.asarray(v).squeeze() for k, v in obs["pixels"].items()}; base_obs = preprocess_observation(obs); ids = [i for i, c in enumerate(r["candidates"]) if c["valid"]]; G = P0.centers(r)[0]; gts, preds = {}, {}
            for k, ci in enumerate(ids):
                c = r["candidates"][ci]; o = {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in base_obs.items()}; o["task"] = [c["instruction"]]; batch = pre(env_pre(o)); sv = prov(pol, batch); info = prov.info; gt = c["anchor"] + off[task]
                src = info["source"]; R["src"][src] = R["src"].get(src, 0) + 1; gts[k] = gt; preds[k] = sv; err = np.linalg.norm(sv.grasp_xyz - gt) * 100; err_v2 = np.linalg.norm(np.array(info["g_v2"]) - gt) * 100; R["err"].append(err); R["err_v2"].append(err_v2); gtC = c["anchor"] + CA[task]; R["errC"].append(np.linalg.norm(sv.grasp_xyz - gtC) * 100); R["errC_v2"].append(np.linalg.norm(np.array(info["g_v2"]) - gtC) * 100)
                pf = info.get("front_props", []); covered = any(np.hypot(p[0] - G[k][0], p[1] - G[k][1]) <= MATCH for p in pf); R["cov"].append(float(covered)); selok = bool(info.get("selected") is not None and np.hypot(pf[info["selected"]][0] - G[k][0], pf[info["selected"]][1] - G[k][1]) <= MATCH); R["ref"].append(float(selok))
                cat = "A_proposal_miss" if not covered else "B_wrong_referent" if not selok else "C_triangulation_failure" if src != "grounded" else "D_anchor_offset_error" if err >= 3 else "success"; R["decomp"][cat] = R["decomp"].get(cat, 0) + 1
                ee0 = np.asarray(base_obs["agent_pos"][0][:3], float) if False else np.asarray(r["state"][:3], float); eul0 = np.asarray(r["state"][3:6], float); pos, rot = build_pregrasp_chunk(ee0, eul0, sv); ok = np.allclose(pos[-1], sv.grasp_xyz) and np.isfinite(pos).all() and np.isfinite(rot).all()
                R["rows"].append({"scene": r["scene_id"], "k": k, "err_cm": err, "err_v2_cm": err_v2, "src": src, "chunk_ok": bool(ok)})
                if k == [j for j, c2 in enumerate([r["candidates"][i] for i in ids]) if c2["is_target"]][0] and nplot < a.plots:
                    nplot += 1; sv2 = StageVars(np.array(info["g_v2"]), sv.grasp_euler, sv.approach_dir, sv.pre_dist, True, np.array(info["g_v2"]) + (sv.wp_xyz - sv.grasp_xyz)); p2, _ = build_pregrasp_chunk(ee0, eul0, sv2); fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                    for j, (ix, iy, nm) in enumerate(((0, 1, "x-y"), (1, 2, "y-z"))):
                        ax[j].plot(p2[:, ix], p2[:, iy], "r-", label="StagePlan-v2 (old)"); ax[j].plot(pos[:, ix], pos[:, iy], "b-", label="v2.1 grounded"); ax[j].plot(*[gt[ix], gt[iy]], "g*", ms=14, label="GT grasp"); ax[j].plot(*[kp[ix], kp[iy]], "k+", ms=10, label="keypoint (historical oracle G)")
                        for ci2 in ids: ax[j].plot(r["candidates"][ci2]["anchor"][ix], r["candidates"][ci2]["anchor"][iy], "o", color="orange", alpha=0.4)
                        ax[j].set_title(f"{r['scene_id']} {nm}"); ax[j].axis("equal")
                    ax[0].legend(fontsize=6); fig.savefig(X / "traj" / f"{r['scene_id']}.png", dpi=90); plt.close(fig)
            for a_ in range(len(ids)):
                for b_ in range(len(ids)):
                    if a_ == b_: continue
                    g = gts[b_] - gts[a_]; ng = np.linalg.norm(g)
                    if ng < 0.03: continue
                    d = preds[b_].grasp_xyz - preds[a_].grasp_xyz; nd = np.linalg.norm(d); R["swap"].append({"cos": float(g @ d / (ng * nd)) if nd > 1e-9 else 0.0, "real": float(d @ g / ng ** 2)})
        env.close(); n = len(R["err"]); out[task] = {"n_instructions": n, "proposal_target_coverage": float(np.mean(R["cov"])), "referent_acc": float(np.mean(R["ref"])), "grasp_error_provider_vs_GT_C(keypoint + measured C-A)": st(R["errC"]), "grasp_error_StagePlan_v2_original_vs_GT_C": st(R["errC_v2"]), "grasp_error_provider_vs_GT_pipeline_labels": st(R["err"]), "grasp_error_StagePlan_v2_original": st(R["err_v2"]), "source_counts": R["src"], "failure_decomposition": R["decomp"],
                     "swap": {"pairs": len(R["swap"]), "cos": float(np.mean([s["cos"] for s in R["swap"]])), "realized": float(np.mean([s["real"] for s in R["swap"]]))}, "chunk_build_ok_frac": float(np.mean([x["chunk_ok"] for x in R["rows"]]))}; print(task, json.dumps(out[task], indent=1), flush=True)
    json.dump(out, open(X / "provider_offline.json", "w"), indent=1)

if __name__ == "__main__": main()
