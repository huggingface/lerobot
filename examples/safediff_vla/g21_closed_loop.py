#!/usr/bin/env python
"""G21 multi-task ORACLE execution test (one fresh OS process per task x arm; localization removed).
ORACLE = the target entity's grasp keypoint `entity.get_grasped_keypoints(physics)[-1]` in the robot-base frame (same definition as the poker G7/G19 oracle): live in PRE_GRASP, frozen at its last pre-grasp value after the reactive close (as in OracleGrounder). It is fed to the decoder through the normal target_point_head slot, so the planner target / reactive-close target are the oracle.
--arm decoder : ORACLE + LEARNED DECODER (unchanged V2 pipeline).
--arm geo     : ORACLE + GEOMETRY. The poker G19 controller, UNCHANGED, no per-task tuning: in PRE_GRASP every chunk's positions := p0 -> oracle+[0,0,8cm] (30 steps) -> oracle (20 steps), constant speed; decoder rotation channels kept; gripper OPEN; close only through the existing reactive close (EE within 3cm of the planner target); POST_GRASP = unchanged learned decoder.
Language: the env wrapper's own native task description (CORRECTED-EVAL path, as G5). GT/oracle never reaches the language path. Per env step: EE, oracle-keypoint distances, phase."""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
from g19_geometry import geo_chunk  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT, robot_base  # noqa: E402
from _multitask_common import RENAME_MAP  # noqa: E402
from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

LOG: list[dict] = []; CHUNKS: list[dict] = []; ST = {"ep": -1, "t": 0}; OPEN_RAW = 1.0


class OracleAnchor:
    def __init__(self, policy): self.policy = policy; self.cache = None; self.last = None

    def anchor(self):
        if self.policy._v2_grasp_phase == 1 and self.last is not None: return self.last
        env = G5.ORACLE_ENV["env"]; physics = env._env.physics; task = env._env.task; kp = np.asarray(task.entities[task.target_entity].get_grasped_keypoints(physics)[-1], dtype=float); self.last = kp - robot_base(env); return self.last

    @torch.no_grad()
    def compute(self, tok, pad, mod, batch) -> dict:
        mean, std = self.policy.action_pos_mean, self.policy.action_pos_std; gt = self.anchor(); return {"xyz_n": ((torch.as_tensor(gt, dtype=std.dtype, device=std.device) - mean) / std).unsqueeze(0), "info": {"oracle_anchor_robot": gt.tolist()}}


@torch.no_grad()
def pre(policy, batch):
    env = G5.ORACLE_ENV["env"]
    if ST["ep"] != G5.ORACLE_ENV["ep"]: ST.update(ep=G5.ORACLE_ENV["ep"], t=0)
    physics = env._env.physics; task = env._env.task; base = robot_base(env); kp = np.asarray(task.entities[task.target_entity].get_grasped_keypoints(physics)[-1], dtype=float); ee_w = np.asarray(task.robot.get_end_effector_pos(physics), dtype=float); d = ee_w - kp
    LOG.append({"ep": ST["ep"], "t": ST["t"], "ee": ee_w - base, "kp": kp - base, "dxy": float(np.linalg.norm(d[:2])), "dz": float(d[2]), "d3": float(np.linalg.norm(d)), "phase": int(policy._v2_grasp_phase), "replaced": False}); ST["t"] += 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--arm", required=True, choices=["decoder", "geo"]); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(3000, 3005))); ap.add_argument("--out", default="outputs/eval/safediff_vla_g21"); a = ap.parse_args()
    out_dir = Path(a.out) / a.task / a.arm; out_dir.mkdir(parents=True, exist_ok=True); train_strings = set(pd.read_parquet(G5.TASKS_PARQUET).index)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); assert policy.config.architecture == "temporal_decoder_grounded_grasp_v2"
    env_cfg = VLABenchEnv(task=a.task); env = make_env(env_cfg, n_envs=1, use_async_envs=False)[a.task][0]
    pre_p, post_p = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}}); env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    grounder = OracleAnchor(policy); G5.GROUNDER["g"] = grounder; policy.target_point_head = G5.OverrideHead(policy.target_point_head, grounder); G5.PRE["fn"] = pre
    inst = G5.install_policy_instrumentation(policy, grounder, pre_p); env_state = G5.install_env_instrumentation(inst["log"], inst["lang"], on_reset=lambda: setattr(grounder, "last", None)); mean, std = policy.action_pos_mean, policy.action_pos_std; gm, gs = policy.action_grip_mean, policy.action_grip_std
    if a.arm == "geo":
        pac = policy.plan_action_chunk
        def geo_pac(batch):
            actions, metrics = pac(batch)
            if policy._v2_grasp_phase != 0 or policy._v2_last_target_xyz is None: return actions, metrics
            ee = (policy._current_state(batch)[0, :3] * policy.state_pos_std + policy.state_pos_mean).cpu().numpy(); tgt = (policy._v2_last_target_xyz[0] * std + mean).cpu().numpy(); g = geo_chunk(ee, tgt); x = actions.clone()
            x[0, :, :3] = (torch.as_tensor(g, dtype=x.dtype, device=x.device) - mean) / std; x[0, :, GRIPPER_INDEX_RAW] = ((OPEN_RAW - gm) / gs).to(x.dtype); LOG[-1]["replaced"] = True; CHUNKS.append({"ep": LOG[-1]["ep"], "t": LOG[-1]["t"], "ee": ee, "target": tgt, "first_off": float(np.linalg.norm(g[0] - ee))}); return x, metrics
        policy.plan_action_chunk = geo_pac
    records: list[dict] = []
    def cb(episode_ix, rollout_data, env_idx, done_index):
        meta = dict(env_state["episode_meta"]); lang = inst["lang"][0] if inst["lang"] else {}; meta["in_training_strings"] = (meta.get("task_description") in train_strings) if meta.get("task_description") else None
        records.append({"episode_ix": episode_ix, "episode_len": done_index + 1, "meta": meta, "language": lang, "steps": list(env_state["steps"])}); inst["log"].clear(); inst["lang"].clear(); print(f"  [{a.task}/{a.arm}] episode {episode_ix} len={done_index + 1} target={meta.get('target_entity')} desc={meta.get('task_description')!r}", flush=True)
    set_seed(a.seeds[0])
    try: info = eval_policy(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post, preprocessor=pre_p, postprocessor=post_p, n_episodes=len(a.seeds), max_episodes_rendered=len(a.seeds), videos_dir=out_dir / "videos", start_seed=a.seeds[0], episode_callback=cb)
    finally:
        env.close(); cls, os_, or_ = env_state["_orig"]; cls.step, cls.reset = os_, or_
    for ep, r in zip(info["per_episode"], records, strict=True): r["seed"] = ep["seed"]; r["success"] = ep["success"]
    (out_dir / "episodes_raw.json").write_text(json.dumps(records)); pickle.dump({"log": LOG, "chunks": CHUNKS}, open(out_dir / "g21_log.pkl", "wb")); print(f"[{a.task}/{a.arm}] successes: {[ep['success'] for ep in info['per_episode']]}")


if __name__ == "__main__":
    main()
