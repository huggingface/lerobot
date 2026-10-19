#!/usr/bin/env python
"""G23 STEP 5 (privileged RELEASE diagnostic, NOT a method): the G21 ORACLE+GEOMETRY arm with ONE change: once the held target object is within 6cm (XY, simulator state) of the goal container centre, the gripper command is forced OPEN (one-shot, for the rest of the episode); every other motion unchanged. Derived from the G23 logging re-run (below).
Original docstring: G23 STEP 2 logging re-run (NOT a new policy: the G21 ORACLE+GEOMETRY arm re-executed deterministically with extra LOGGING only; outcomes are checked against the G21 episodes). Adds, per env step: EE, target-object position, goal-container position (toy giftbox_seen / fruit plate_seen / mahjong placemat), phase, executed gripper command; per replan: the decoder's RAW gripper/xyz chunk (G5's plan_action_chunk wrapper, i.e. BEFORE the reactive-close override and before the geometric PRE_GRASP replacement). The G21 logs did not contain these."""
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

CONTAINER = {"select_toy": "giftbox_seen", "select_fruit": "plate_seen", "select_mahjong": "placemat", "select_poker": None}; RELEVENTS: list[dict] = []; STEPS: list[dict] = []; RE: list[dict] = []; ST = {"ep": -1, "t": 0}


@torch.no_grad()
def pre(policy, batch):
    G21.pre(policy, batch); env = G5.ORACLE_ENV["env"]; e = env._env; base = robot_base(env); cn = CONTAINER[TASK]; cpos = (np.asarray(e.task.entities[cn].get_xpos(e.physics), float) - base) if cn else None; tpos = np.asarray(e.task.entities[e.task.target_entity].get_xpos(e.physics), float) - base
    STEPS.append({"ep": G21.LOG[-1]["ep"], "t": G21.LOG[-1]["t"], "ee": G21.LOG[-1]["ee"], "obj": tpos, "cont": cpos, "phase": int(policy._v2_grasp_phase), "grasped": bool(e.task.entities[e.task.target_entity].is_grasped(e.physics, e.task.robot))})


def main():
    global TASK
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(3000, 3005))); ap.add_argument("--out", default="outputs/eval/safediff_vla_g23/release_diag"); a = ap.parse_args(); TASK = a.task
    out_dir = Path(a.out) / a.task; out_dir.mkdir(parents=True, exist_ok=True); train_strings = set(pd.read_parquet(G5.TASKS_PARQUET).index)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); env_cfg = VLABenchEnv(task=a.task); env = make_env(env_cfg, n_envs=1, use_async_envs=False)[a.task][0]
    pre_p, post_p = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}}); env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    grounder = G21.OracleAnchor(policy); G5.GROUNDER["g"] = grounder; policy.target_point_head = G5.OverrideHead(policy.target_point_head, grounder); G5.PRE["fn"] = pre
    inst = G5.install_policy_instrumentation(policy, grounder, pre_p); env_state = G5.install_env_instrumentation(inst["log"], inst["lang"], on_reset=lambda: setattr(grounder, "last", None)); mean, std = policy.action_pos_mean, policy.action_pos_std; gm, gs = policy.action_grip_mean, policy.action_grip_std; pac = policy.plan_action_chunk
    def geo_pac(batch):
        actions, metrics = pac(batch); raw = G5.CHUNK["last"]  # decoder raw chunk [H,7] normalised (recorded by G5's wrapper)
        RE.append({"ep": G21.LOG[-1]["ep"], "t": G21.LOG[-1]["t"], "phase": int(policy._v2_grasp_phase), "grip": (raw[:, GRIPPER_INDEX_RAW] * gs + gm).cpu().numpy(), "xyz": (raw[:, :3] * std + mean).cpu().numpy()})
        if policy._v2_grasp_phase != 0 or policy._v2_last_target_xyz is None: return actions, metrics
        ee = (policy._current_state(batch)[0, :3] * policy.state_pos_std + policy.state_pos_mean).cpu().numpy(); tgt = (policy._v2_last_target_xyz[0] * std + mean).cpu().numpy(); g = geo_chunk(ee, tgt); x = actions.clone()
        x[0, :, :3] = (torch.as_tensor(g, dtype=x.dtype, device=x.device) - mean) / std; x[0, :, GRIPPER_INDEX_RAW] = ((G21.OPEN_RAW - gm) / gs).to(x.dtype); return x, metrics
    policy.plan_action_chunk = geo_pac; RELEASE = {"on": False, "t": None}; sel0 = policy.select_action; open_n = (G21.OPEN_RAW - gm) / gs
    def sel(batch):
        act = sel0(batch); env_ = G5.ORACLE_ENV["env"]; e_ = env_._env; cn = CONTAINER[TASK]
        if G21.LOG and G21.LOG[-1]["t"] == 0: RELEASE.update(on=False, t=None)
        if not RELEASE["on"] and cn and policy._v2_grasp_phase == 1:
            b_ = robot_base(env_); o = np.asarray(e_.task.entities[e_.task.target_entity].get_xpos(e_.physics), float) - b_; c = np.asarray(e_.task.entities[cn].get_xpos(e_.physics), float) - b_
            if np.linalg.norm(o[:2] - c[:2]) < 0.06 and e_.task.entities[e_.task.target_entity].is_grasped(e_.physics, e_.task.robot): RELEASE.update(on=True, t=G21.LOG[-1]["t"]); RELEVENTS.append({"ep": G21.LOG[-1]["ep"], "t": RELEASE["t"]})
        if RELEASE["on"]: act = act.clone(); act[..., GRIPPER_INDEX_RAW] = torch.as_tensor(open_n, dtype=act.dtype, device=act.device)
        return act
    policy.select_action = sel; records: list[dict] = []
    def cb(episode_ix, rollout_data, env_idx, done_index):
        meta = dict(env_state["episode_meta"]); records.append({"episode_ix": episode_ix, "episode_len": done_index + 1, "meta": meta, "steps": list(env_state["steps"])}); inst["log"].clear(); inst["lang"].clear(); print(f"  [{a.task}] episode {episode_ix} len={done_index + 1}", flush=True)
    set_seed(a.seeds[0])
    try: info = eval_policy(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post, preprocessor=pre_p, postprocessor=post_p, n_episodes=len(a.seeds), max_episodes_rendered=len(a.seeds), videos_dir=out_dir / "videos", start_seed=a.seeds[0], episode_callback=cb)
    finally:
        env.close(); cls, os_, or_ = env_state["_orig"]; cls.step, cls.reset = os_, or_
    for ep, r in zip(info["per_episode"], records, strict=True): r["seed"] = ep["seed"]; r["success"] = ep["success"]
    pickle.dump({"release_events": RELEVENTS, "steps": STEPS, "replans": RE, "episodes": [{"seed": r["seed"], "success": r["success"], "len": r["episode_len"], "close": next((i for i, s in enumerate(r["steps"]) if s.get("reactive_close_event")), None), "cmd_grip": [s["commanded_gripper_raw"] for s in r["steps"]]} for r in records]}, open(out_dir / "g23_release_log.pkl", "wb")); print(f"[{a.task}] successes: {[ep['success'] for ep in info['per_episode']]}")


if __name__ == "__main__":
    main()
