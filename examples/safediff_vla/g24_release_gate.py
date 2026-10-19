#!/usr/bin/env python
"""G24 INSTRUCTION-CONDITIONED RELEASE GATE (toy, fruit, mahjong; seeds 3000-3004; one fresh process per task; no learning, no timing sweep).
Base = the G21 ORACLE+GEOMETRY arm (oracle grasp keypoint, G19 geometric PRE_GRASP chunk, reactive close at 3cm, POST_GRASP = learned decoder, native task description). The ONLY change is how the executed gripper command is formed after the reactive close (which is otherwise a one-shot latch forcing CLOSED for the rest of the episode):
   if not release_ready: executed gripper = CLOSED (any decoder OPEN suppressed)      else: executed gripper = the DECODER'S OWN raw gripper command (passed through unchanged).
 release_ready = goal entity found AND object-goal XY distance < 6cm (one tau, no sweep; point distance to the container centre because no region-inside test is exposed) AND (target.is_grasped() OR release already triggered earlier in this episode, so the reopening itself does not re-close the gate; the gate closes again if the object leaves the 6cm disc).  Never based on EE position.
 Goal entity = the container whose name appears in the instruction string (toy 'giftbox_seen', fruit 'plate_seen'); mahjong instructions name none -> task-defined destination 'placemat' (assumption; flagged 'task-config').
Per step: decoder RAW gripper command (before override), executed command, release_ready, is_grasped, object/container/EE positions, phase."""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
import g21_closed_loop as G21  # noqa: E402
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

TAU = 0.06; FALLBACK = {"select_mahjong": "placemat"}; STEPS: list[dict] = []; GATE = {"triggered": False, "goal": None, "src": None, "ep": -1}; EPS_GOAL: dict = {}


def find_goal(task_name, e):
    desc = getattr(G5.ORACLE_ENV["env"], "task_description", "") or ""; names = [n for n in e.task.entities if n in desc and n != e.task.target_entity and not any(n == k for k in ("table",))]
    if names: return names[0], "instruction"
    return FALLBACK.get(task_name), "task-config"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(3000, 3005))); ap.add_argument("--out", default="outputs/eval/safediff_vla_g24"); a = ap.parse_args()
    out_dir = Path(a.out) / a.task; out_dir.mkdir(parents=True, exist_ok=True); train_strings = set(pd.read_parquet(G5.TASKS_PARQUET).index)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); env_cfg = VLABenchEnv(task=a.task); env = make_env(env_cfg, n_envs=1, use_async_envs=False)[a.task][0]
    pre_p, post_p = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}}); env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    grounder = G21.OracleAnchor(policy); G5.GROUNDER["g"] = grounder; policy.target_point_head = G5.OverrideHead(policy.target_point_head, grounder); G5.PRE["fn"] = G21.pre
    inst = G5.install_policy_instrumentation(policy, grounder, pre_p)
    def on_reset(): grounder.last = None; GATE.update(triggered=False, goal=None, src=None, ep=GATE["ep"] + 1)
    env_state = G5.install_env_instrumentation(inst["log"], inst["lang"], on_reset=on_reset); mean, std = policy.action_pos_mean, policy.action_pos_std; gm, gs = policy.action_grip_mean, policy.action_grip_std; pac = policy.plan_action_chunk
    def geo_pac(batch):
        actions, metrics = pac(batch)
        if policy._v2_grasp_phase != 0 or policy._v2_last_target_xyz is None: return actions, metrics
        ee = (policy._current_state(batch)[0, :3] * policy.state_pos_std + policy.state_pos_mean).cpu().numpy(); tgt = (policy._v2_last_target_xyz[0] * std + mean).cpu().numpy(); g = geo_chunk(ee, tgt); x = actions.clone()
        x[0, :, :3] = (torch.as_tensor(g, dtype=x.dtype, device=x.device) - mean) / std; x[0, :, GRIPPER_INDEX_RAW] = ((G21.OPEN_RAW - gm) / gs).to(x.dtype); return x, metrics
    policy.plan_action_chunk = geo_pac; orig_rc = policy._grounded_grasp_v2_reactive_close
    def gated(action, current_state):
        raw = action[..., GRIPPER_INDEX_RAW].clone(); was = policy._v2_close_triggered; out = orig_rc(action, current_state)
        env_ = G5.ORACLE_ENV["env"]; e_ = env_._env; base = robot_base(env_); obj_e = e_.task.entities[e_.task.target_entity]; obj = np.asarray(obj_e.get_xpos(e_.physics), float) - base; ready = False; dxy = float("nan"); grasped = bool(obj_e.is_grasped(e_.physics, e_.task.robot))
        if GATE["goal"] is None: GATE["goal"], GATE["src"] = find_goal(a.task, e_); EPS_GOAL[GATE["ep"]] = (GATE["goal"], GATE["src"])
        cont = None
        if GATE["goal"] is not None and GATE["goal"] in e_.task.entities:
            cont = np.asarray(e_.task.entities[GATE["goal"]].get_xpos(e_.physics), float) - base; dxy = float(np.linalg.norm(obj[:2] - cont[:2])); near = dxy < TAU
            if policy._v2_close_triggered and was:
                if near and (grasped or GATE["triggered"]): ready = True; GATE["triggered"] = True
        if policy._v2_close_triggered and was and ready: out = out.clone(); out[..., GRIPPER_INDEX_RAW] = raw
        STEPS.append({"ep": GATE["ep"], "t": G21.LOG[-1]["t"] if G21.LOG else -1, "raw": float(raw.reshape(-1)[0] * gs + gm), "exec": float(out[..., GRIPPER_INDEX_RAW].reshape(-1)[0] * gs + gm), "ready": ready, "grasped": grasped, "dxy_goal": dxy, "obj": obj, "cont": cont, "ee": G21.LOG[-1]["ee"] if G21.LOG else None, "phase": int(policy._v2_grasp_phase), "post": bool(policy._v2_close_triggered and was)})
        return out
    policy._grounded_grasp_v2_reactive_close = gated; records: list[dict] = []
    def cb(episode_ix, rollout_data, env_idx, done_index):
        meta = dict(env_state["episode_meta"]); records.append({"episode_ix": episode_ix, "episode_len": done_index + 1, "meta": meta, "steps": [{"target_pos": s["target_pos"], "is_grasped": s["is_grasped"], "reactive_close_event": s.get("reactive_close_event")} for s in env_state["steps"]]}); inst["log"].clear(); inst["lang"].clear(); print(f"  [{a.task}] episode {episode_ix} len={done_index + 1} goal={EPS_GOAL.get(episode_ix)}", flush=True)
    set_seed(a.seeds[0])
    try: info = eval_policy(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post, preprocessor=pre_p, postprocessor=post_p, n_episodes=len(a.seeds), max_episodes_rendered=len(a.seeds), videos_dir=out_dir / "videos", start_seed=a.seeds[0], episode_callback=cb)
    finally:
        env.close(); cls, os_, or_ = env_state["_orig"]; cls.step, cls.reset = os_, or_
    for ep, r in zip(info["per_episode"], records, strict=True): r["seed"] = ep["seed"]; r["success"] = ep["success"]
    pickle.dump({"steps": STEPS, "goal": EPS_GOAL, "episodes": [{"seed": r["seed"], "success": r["success"], "len": r["episode_len"], "close": next((i for i, s in enumerate(r["steps"]) if s["reactive_close_event"]), None), "tpos": [s["target_pos"] for s in r["steps"]]} for r in records]}, open(out_dir / "g24_log.pkl", "wb")); print(f"[{a.task}] successes: {[ep['success'] for ep in info['per_episode']]}")


if __name__ == "__main__":
    main()
