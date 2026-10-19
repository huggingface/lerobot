#!/usr/bin/env python
"""G28 BOOK / DRINK global-vs-wrist arms (one fresh process per task x arm; oracle grasp keypoint into the planner-target slot; latched reactive close at 3cm; POST_GRASP = unchanged learned decoder; native task description). Only the PRE_GRASP approach source changes:
 A  GLOBAL-POINT    : G21 controller (p0 -> keypoint+8cm world-z -> keypoint), decoder orientation.
 B  GLOBAL-POSE     : G22-A (expert grasp pose euler(-pi/2,-pi/2,0), approach (0,1,0); p0 -> p_grasp-8cm*d -> p_grasp, 30+20 steps, orientation slerp).
 C  GLOBAL-WAYPOINT : expert-derived target-relative route: p0 -> w25 (median expert EE offset at 25cm from the keypoint) -> expert prepare point (keypoint - 10cm*d) -> p_grasp, 20+15+15 steps, orientation slerp over the first 35 steps; open-loop, no wrist.
 D  WRIST-LOCAL     : as C until the EE is within 2cm of the prepare point, then a per-step visual servo for the last approach only: command = EE + 0.7cm*d + lateral correction; correction from the WRIST view (live projection of the keypoint into the wrist camera, gated by target visibility >20px), error = live (u,v) - expert reference (u,v)(EE-keypoint distance), lateral world step = 0.5*(depth/f)*(e_u*cam_x - e_v*cam_y) minus its component along d, clipped at 1cm/step (ONE fixed gain); orientation held at the expert grasp orientation.
 D2 FROZEN-WRIST    : as D but the wrist error is read ONCE at the start of the local stage and applied open-loop (full correction spread over 10 steps).
The servo integrates its command (P += 0.7cm*d + correction, P initialised at the EE when the local stage starts) so that a steady tracking sag is removed by the image error rather than accumulating.
Per step: physics snapshot (g25_common), wrist/front target projection + visible pixels, small wrist/front frames every 5 steps; non-target displacement is computed from the env records."""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
import g21_closed_loop as G21  # noqa: E402
import g22_closed_loop as G22  # noqa: E402
import g25_common as C25  # noqa: E402
import g28_common as C28  # noqa: E402
from g19_geometry import geo_chunk  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT, robot_base  # noqa: E402
from _multitask_common import RENAME_MAP  # noqa: E402
from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW, ROTATION_SLICE_RAW  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

STEPS: list[dict] = []; FRAMES: dict = {}; CH: list[dict] = []; LOC = {"on": False, "t0": None, "e0": None, "n": 0, "ep": -1, "P": None}; GAIN = 0.5; ADV = 0.007; CLIP = 0.01; F_PX = 240 / np.tan(np.deg2rad(22.5)); FWD = np.array([0.0, 1.0, 0.0])


def chunk_C(ee, eul, g, w25, d, quat):
    wp = g + np.asarray(w25); pre_ = g - 0.10 * d; seg = [(wp, 20), (pre_, 15), (g, 15)]; pts = []; cur = ee
    for tgt, n in seg: a = (np.arange(1, n + 1) / n)[:, None]; pts.append(cur + a * (tgt - cur)); cur = tgt
    pos = np.concatenate(pts, 0); r0 = Rotation.from_euler("xyz", eul); r1 = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]); sl = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1])); fr = np.concatenate([np.arange(1, 36) / 35, np.ones(15)]); e2 = sl(fr).as_euler("xyz")
    return pos, np.unwrap(np.concatenate([eul[None], e2], 0), axis=0)[1:]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--arm", required=True, choices=["A", "B", "C", "D", "D2"]); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(3000, 3005))); ap.add_argument("--out", default="outputs/eval/safediff_vla_g28"); a = ap.parse_args()
    out_dir = Path(a.out) / a.task / a.arm; out_dir.mkdir(parents=True, exist_ok=True); refs = json.load(open(Path(a.out) / "expert_refs.json"))[a.task]; w25 = np.array(refs["w25_rel"]); rb = sorted((int(k), np.array(v)) for k, v in refs["ref"].items()); train_strings = set(pd.read_parquet(G5.TASKS_PARQUET).index)
    G22.KEY_EULER = (-np.pi / 2, -np.pi / 2, 0.0)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); env_cfg = VLABenchEnv(task=a.task); env = make_env(env_cfg, n_envs=1, use_async_envs=False)[a.task][0]
    pre_p, post_p = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}}); env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    grounder = G21.OracleAnchor(policy); G5.GROUNDER["g"] = grounder; policy.target_point_head = G5.OverrideHead(policy.target_point_head, grounder)
    def pre(policy_, batch):
        G21.pre(policy_, batch); env_ = G5.ORACLE_ENV["env"]; e = env_._env; s = C25.snapshot(e); ph = e.physics; kpw = np.array(e.task.entities[e.task.target_entity].get_grasped_keypoints(ph)[0], float, copy=True); base = robot_base(env_)
        s.update({"ep": G21.LOG[-1]["ep"], "t": G21.LOG[-1]["t"], "ee": G21.LOG[-1]["ee"], "kp": G21.LOG[-1]["kp"], "phase": int(policy_._v2_grasp_phase), "local": bool(LOC["on"]), "wrist_uv": C28.cam_proj(ph, C28.WRIST, kpw), "wrist_vis": C28.target_pixels(e, C28.WRIST)[0], "front_vis": C28.target_pixels(e, C28.FRONT)[0]})
        STEPS.append(s)
        if s["t"] % 5 == 0: FRAMES[(s["ep"], s["t"])] = (C28.frame(e, C28.WRIST, 120), C28.frame(e, C28.FRONT, 120))
        if s["t"] == 0: LOC.update(on=False, t0=None, e0=None, n=0)
    G5.PRE["fn"] = pre
    def on_reset(): grounder.last = None; G22.pose_for_episode(); LOC.update(on=False, t0=None, e0=None, n=0)
    inst = G5.install_policy_instrumentation(policy, grounder, pre_p); env_state = G5.install_env_instrumentation(inst["log"], inst["lang"], on_reset=on_reset); mean, std = policy.action_pos_mean, policy.action_pos_std; gm, gs = policy.action_grip_mean, policy.action_grip_std; pac = policy.plan_action_chunk
    def pac2(batch):
        actions, metrics = pac(batch)
        if policy._v2_grasp_phase != 0 or policy._v2_last_target_xyz is None: return actions, metrics
        cs = policy._current_state(batch)[0]; ee = (cs[:3] * policy.state_pos_std + policy.state_pos_mean).cpu().numpy(); eul = (cs[3:6] * policy.state_rot_std + policy.state_rot_mean).cpu().numpy().astype(float); g = (policy._v2_last_target_xyz[0] * std + mean).cpu().numpy(); x = actions.clone(); dt, dv = x.dtype, x.device
        if a.arm == "A": pos = geo_chunk(ee, g); rot = None
        elif a.arm == "B": pos, rot = G22.pose_chunk(ee, eul, g)
        else: pos, rot = chunk_C(ee, eul, g, w25, G22.POSE["d"], G22.POSE["quat"])
        x[0, :, :3] = (torch.as_tensor(pos, dtype=dt, device=dv) - mean) / std
        if rot is not None: x[0, :, ROTATION_SLICE_RAW] = (torch.as_tensor(rot, dtype=dt, device=dv) - policy.action_rot_mean) / policy.action_rot_std
        x[0, :, GRIPPER_INDEX_RAW] = ((G21.OPEN_RAW - gm) / gs).to(dt); CH.append({"ep": G21.LOG[-1]["ep"], "t": G21.LOG[-1]["t"], "first_off": float(np.linalg.norm(pos[0] - ee))}); return x, metrics
    policy.plan_action_chunk = pac2; sel0 = policy.select_action
    def ref_uv(dist_cm):
        xs = np.array([b for b, _ in rb], float); vs = np.array([v for _, v in rb]); return np.array([np.interp(dist_cm, xs, vs[:, i]) for i in range(3)])
    def sel(batch):
        act = sel0(batch)
        if a.arm not in ("D", "D2") or policy._v2_grasp_phase != 0: return act
        env_ = G5.ORACLE_ENV["env"]; e = env_._env; L = G21.LOG[-1]; ee = L["ee"]; kp = L["kp"]; d = G22.POSE["d"]; prep = kp - 0.10 * d
        if not LOC["on"] and np.linalg.norm(ee - prep) < 0.02: LOC.update(on=True, t0=L["t"], e0=None, n=0, P=ee.copy())
        if not LOC["on"]: return act
        ph = e.physics; base = robot_base(env_); kpw = kp + base; uv = C28.cam_proj(ph, C28.WRIST, kpw); dist = float(np.linalg.norm(ee - kp)); rf = ref_uv(dist * 100); vis = C28.target_pixels(e, C28.WRIST)[0]; corr = np.zeros(3)
        Rm = np.array(ph.data.cam_xmat[C28.WRIST]).reshape(3, 3)
        def lateral(gain):
            if uv is None or vis <= 20: return np.zeros(3)
            eu, ev = uv[0] - rf[0], uv[1] - rf[1]; v = gain * (rf[2] / F_PX) * (eu * Rm[:, 0] - ev * Rm[:, 1]); v = v - (v @ d) * d; n = np.linalg.norm(v); return v if n <= CLIP or n == 0 else v * CLIP / n
        if a.arm == "D": corr = lateral(GAIN)
        else:
            if LOC["e0"] is None: LOC["e0"] = lateral(1.0) * 0 + (lateral(1.0) if True else 0)
            corr = LOC["e0"] / 10.0 if LOC["n"] < 10 else np.zeros(3)
        LOC["n"] += 1; LOC["P"] = LOC["P"] + ADV * d + corr; cmd = LOC["P"].copy(); act = act.clone(); dt, dv = act.dtype, act.device
        act[..., :3] = ((torch.as_tensor(cmd, dtype=dt, device=dv) - mean) / std); eul = Rotation.from_quat([G22.POSE["quat"][1], G22.POSE["quat"][2], G22.POSE["quat"][3], G22.POSE["quat"][0]]).as_euler("xyz")
        cs0 = policy._current_state(batch)[0]; e_cur = (cs0[3:6] * policy.state_rot_std + policy.state_rot_mean).cpu().numpy().astype(float); eul = e_cur + (np.unwrap(np.array([e_cur, eul]), axis=0)[1] - e_cur)
        act[..., ROTATION_SLICE_RAW] = ((torch.as_tensor(eul, dtype=dt, device=dv) - policy.action_rot_mean) / policy.action_rot_std); STEPS[-1].update({"servo_corr": corr.copy(), "cmd_minus_ee_cm": (cmd - ee) * 100, "servo_corr_cm": float(np.linalg.norm(corr) * 100), "servo_e_px": None if uv is None else float(np.hypot(uv[0] - rf[0], uv[1] - rf[1])), "servo_vis": int(vis)}); return act
    policy.select_action = sel; records: list[dict] = []
    def cb(episode_ix, rollout_data, env_idx, done_index):
        meta = dict(env_state["episode_meta"]); records.append({"episode_ix": episode_ix, "episode_len": done_index + 1, "meta": meta, "steps": [{"target_pos": s["target_pos"], "is_grasped": s["is_grasped"], "reactive_close_event": s.get("reactive_close_event"), "candidates_world": s["candidates_world"]} for s in env_state["steps"]]}); inst["log"].clear(); inst["lang"].clear(); print(f"  [{a.task}/{a.arm}] episode {episode_ix} len={done_index + 1} target={meta.get('target_entity')}", flush=True)
    set_seed(a.seeds[0])
    try: info = eval_policy(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post, preprocessor=pre_p, postprocessor=post_p, n_episodes=len(a.seeds), max_episodes_rendered=len(a.seeds), videos_dir=out_dir / "videos", start_seed=a.seeds[0], episode_callback=cb)
    finally:
        env.close(); cls, os_, or_ = env_state["_orig"]; cls.step, cls.reset = os_, or_
    for ep, r in zip(info["per_episode"], records, strict=True): r["seed"] = ep["seed"]; r["success"] = ep["success"]
    pickle.dump({"steps": STEPS, "chunks": CH, "episodes": [{"seed": r["seed"], "success": r["success"], "len": r["episode_len"], "target": r["meta"].get("target_entity"), "rec": r["steps"]} for r in records]}, open(out_dir / "g28_trace.pkl", "wb")); pickle.dump(FRAMES, open(out_dir / "g28_frames.pkl", "wb")); print(f"[{a.task}/{a.arm}] successes: {[ep['success'] for ep in info['per_episode']]}")


if __name__ == "__main__":
    main()
