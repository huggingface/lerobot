#!/usr/bin/env python
"""G22-A ORACLE GRASP-POSE execution (book, drink; seeds 3000-3004; one fresh process per task). Same pipeline as G21 (oracle keypoint into the planner-target slot, reactive close at 3cm, POST_GRASP = unchanged learned decoder, native task description), but the PRE_GRASP chunk is built from the EXPERT GRASP POSE instead of a world-z waypoint:
 p_grasp = entity.get_grasped_keypoints(physics)[0] (live, = planner target) | R_grasp = the expert primitive's own grasp orientation for these tasks, euler_to_quaternion(-pi/2,-pi/2,0) (`SkillLib.pick(prior_eulers=[[-pi/2,-pi/2,0]])` in select_book/select_drink_series) | d_approach = the expert's own approach direction, the gripper forward vector VLABench's `find_keypoint_and_prepare_grasp` uses (`env.robot.gripper_pcd(keypoint, key_quat)`; the expert's prepare point is keypoint - 0.1*d_approach).
 NOTE (deviation from 'EE trajectory K steps before close'): the live scene has no recorded demonstration, so the expert's planner outputs are used directly; they are the exact same quantities the demonstrations were generated from. h = 8cm (fixed, NOT the expert's 10cm), no K.
 chunk: p0 -> p_grasp - 8cm*d_approach (30 steps) -> p_grasp (20 steps), constant speed; orientation: slerp(current EE orientation -> R_grasp) over the first 30 steps, then held; gripper OPEN; close = existing reactive close (EE within 3cm of the planner target)."""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
import g21_closed_loop as G21  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from _multitask_common import RENAME_MAP  # noqa: E402
from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW, ROTATION_SLICE_RAW  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from VLABench.utils.utils import euler_to_quaternion  # noqa: E402

H, NA, NB = 0.08, 30, 20; POSE: dict = {}; CHUNKS: list[dict] = []; KEY_EULER = (-np.pi / 2, -np.pi / 2, 0.0)


def pose_for_episode():
    env = G5.ORACLE_ENV["env"]; e = env._env; ent = e.task.entities[e.task.target_entity]; kp = np.asarray(ent.get_grasped_keypoints(e.physics)[0], float); q = euler_to_quaternion(*KEY_EULER); _, fwd = e.robot.gripper_pcd(kp, q); POSE.update(quat=np.asarray(q, float), d=np.asarray(fwd, float) / np.linalg.norm(fwd))


def pose_chunk(ee, euler_cur, g):
    d = POSE["d"]; pre_ = g - H * d; a = (np.arange(1, NA + 1) / NA)[:, None]; b = (np.arange(1, NB + 1) / NB)[:, None]; pos = np.concatenate([ee + a * (pre_ - ee), pre_ + b * (g - pre_)], 0)
    q = POSE["quat"]; r0 = Rotation.from_euler("xyz", euler_cur); r1 = Rotation.from_quat([q[1], q[2], q[3], q[0]]); sl = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1])); fr = np.concatenate([np.arange(1, NA + 1) / NA, np.ones(NB)]); eul = sl(fr).as_euler("xyz")
    full = np.unwrap(np.concatenate([euler_cur[None], eul], 0), axis=0)[1:]; return pos, full


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(3000, 3005))); ap.add_argument("--out", default="outputs/eval/safediff_vla_g22"); a = ap.parse_args()
    out_dir = Path(a.out) / a.task / "pose"; out_dir.mkdir(parents=True, exist_ok=True); train_strings = set(pd.read_parquet(G5.TASKS_PARQUET).index)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); env_cfg = VLABenchEnv(task=a.task); env = make_env(env_cfg, n_envs=1, use_async_envs=False)[a.task][0]
    pre_p, post_p = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}}); env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    grounder = G21.OracleAnchor(policy); G5.GROUNDER["g"] = grounder; policy.target_point_head = G5.OverrideHead(policy.target_point_head, grounder); G5.PRE["fn"] = G21.pre
    inst = G5.install_policy_instrumentation(policy, grounder, pre_p)
    def on_reset(): grounder.last = None; pose_for_episode()
    env_state = G5.install_env_instrumentation(inst["log"], inst["lang"], on_reset=on_reset); mean, std = policy.action_pos_mean, policy.action_pos_std; gm, gs = policy.action_grip_mean, policy.action_grip_std; pac = policy.plan_action_chunk
    def pose_pac(batch):
        actions, metrics = pac(batch)
        if policy._v2_grasp_phase != 0 or policy._v2_last_target_xyz is None: return actions, metrics
        cs = policy._current_state(batch)[0]; ee = (cs[:3] * policy.state_pos_std + policy.state_pos_mean).cpu().numpy(); eul = (cs[3:6] * policy.state_rot_std + policy.state_rot_mean).cpu().numpy().astype(float); g = (policy._v2_last_target_xyz[0] * std + mean).cpu().numpy(); pos, rot = pose_chunk(ee, eul, g); x = actions.clone(); dt = x.dtype; dv = x.device
        x[0, :, :3] = (torch.as_tensor(pos, dtype=dt, device=dv) - mean) / std; x[0, :, ROTATION_SLICE_RAW] = (torch.as_tensor(rot, dtype=dt, device=dv) - policy.action_rot_mean) / policy.action_rot_std; x[0, :, GRIPPER_INDEX_RAW] = ((G21.OPEN_RAW - gm) / gs).to(dt)
        G21.LOG[-1]["replaced"] = True; CHUNKS.append({"ep": G21.LOG[-1]["ep"], "t": G21.LOG[-1]["t"], "ee": ee, "target": g, "first_off": float(np.linalg.norm(pos[0] - ee)), "rot_first_deg": float(np.degrees(np.linalg.norm(rot[0] - eul)))}); return x, metrics
    policy.plan_action_chunk = pose_pac; records: list[dict] = []
    def cb(episode_ix, rollout_data, env_idx, done_index):
        meta = dict(env_state["episode_meta"]); lang = inst["lang"][0] if inst["lang"] else {}; meta["in_training_strings"] = (meta.get("task_description") in train_strings) if meta.get("task_description") else None
        records.append({"episode_ix": episode_ix, "episode_len": done_index + 1, "meta": meta, "language": lang, "steps": list(env_state["steps"])}); inst["log"].clear(); inst["lang"].clear(); print(f"  [{a.task}/pose] episode {episode_ix} len={done_index + 1} target={meta.get('target_entity')} desc={meta.get('task_description')!r}", flush=True)
    set_seed(a.seeds[0])
    try: info = eval_policy(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post, preprocessor=pre_p, postprocessor=post_p, n_episodes=len(a.seeds), max_episodes_rendered=len(a.seeds), videos_dir=out_dir / "videos", start_seed=a.seeds[0], episode_callback=cb)
    finally:
        env.close(); cls, os_, or_ = env_state["_orig"]; cls.step, cls.reset = os_, or_
    for ep, r in zip(info["per_episode"], records, strict=True): r["seed"] = ep["seed"]; r["success"] = ep["success"]
    (out_dir / "episodes_raw.json").write_text(json.dumps(records)); pickle.dump({"log": G21.LOG, "chunks": CHUNKS}, open(out_dir / "g21_log.pkl", "wb")); print(f"[{a.task}/pose] successes: {[ep['success'] for ep in info['per_episode']]}")


if __name__ == "__main__":
    main()
