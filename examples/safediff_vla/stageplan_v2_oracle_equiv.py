#!/usr/bin/env python
"""StagePlan-v2 HISTORICAL-EQUIVALENCE test: the NEW execution pipeline (`SafeDiffVLAStagePlanV2Policy`) driven by the exact ORACLE stage
variables the G19-G30 diagnostics used. PRIVILEGED harness (simulator keypoints / goal poses; task name allowed here only, as in the
historical diagnostics) -- NOT an inference path. One task, several seeds, one process.

oracle variables per task (historical code):
  poker/mahjong/fruit/toy (G19/G21/G24): G = keypoint[0], d = -z, r = 8 cm, no waypoint, R = family expert grasp rotation (TRAIN labels)
  book (G22/G29)                       : G = keypoint[0], R = euler(-pi/2,-pi/2,0), d = gripper forward vector of R, r = 8 cm
  drink (G28 C)                        : as book + global waypoint W = G + w25_rel (expert_refs.json), r = 10 cm
release: --release latch (G21), oracle_gate (G24; mahjong/fruit/toy), passthrough (decoder OPEN, no gate)
Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/stageplan_v2_oracle_equiv.py --task select_poker --release latch
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
import g25_common as C25  # noqa: E402
from _multitask_common import RENAME_MAP  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT, robot_base  # noqa: E402

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.vlabench import VLABenchEnv as Impl  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.stageplan_v2 import StageVars, SafeDiffVLAStagePlanV2Policy  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

GOAL = {"select_fruit": "plate_seen", "select_toy": "giftbox_seen", "select_mahjong": "placemat"}
POSE_TASKS = ("select_book", "select_drink")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[3000, 3001, 3002, 3003, 3004])
    ap.add_argument("--release", default="latch", choices=["latch", "oracle_gate", "passthrough"]); ap.add_argument("--out", default="outputs/stageplan_v2/oracle_equiv"); a = ap.parse_args()
    out = Path(a.out) / a.release / a.task; out.mkdir(parents=True, exist_ok=True)
    labels = pickle.load(open("outputs/stageplan/labels.pkl", "rb")); refs = json.load(open("outputs/eval/safediff_vla_g28/expert_refs.json"))
    fam = [l["P1"][3:] for l in labels.values() if l["valid"] and l["fam"] == a.task]; R_fam = Rotation.from_euler("xyz", np.array(fam)).mean().as_euler("xyz")
    pol = SafeDiffVLAStagePlanV2Policy.from_pretrained(CHECKPOINT).to("cuda").eval(); env_cfg = VLABenchEnv(task=a.task); env = make_env(env_cfg, n_envs=1, use_async_envs=False)[a.task][0]
    H = {"e": None, "triggered": False}

    def provider(policy, batch) -> StageVars:  # PRIVILEGED
        e = env.unwrapped.envs[0]; d = e._env; t = d.task; ent = t.entities[t.target_entity]; base = robot_base(e)
        kp = np.asarray(ent.get_grasped_keypoints(d.physics)[0], float) - base
        if a.task in POSE_TASKS:
            from VLABench.utils.utils import euler_to_quaternion
            q = euler_to_quaternion(-np.pi / 2, -np.pi / 2, 0.0); _, fwd = d.robot.gripper_pcd(kp + base, q); dvec = np.asarray(fwd, float) / np.linalg.norm(fwd); eul = np.array([-np.pi / 2, -np.pi / 2, 0.0])
        else: dvec, eul = np.array([0.0, 0.0, -1.0]), R_fam
        if a.task == "select_drink": return StageVars(kp, eul, dvec, 0.10, True, kp + np.array(refs["select_drink"]["w25_rel"]))
        return StageVars(kp, eul, dvec, 0.08, False, None)

    def gate() -> bool:  # PRIVILEGED, historical G24
        e = env.unwrapped.envs[0]; d = e._env; t = d.task; base = robot_base(e); ent = t.entities[t.target_entity]
        goal = GOAL.get(a.task)
        if goal is None or goal not in t.entities: return False
        obj = np.asarray(ent.get_xpos(d.physics), float) - base; cont = np.asarray(t.entities[goal].get_xpos(d.physics), float) - base
        near = float(np.linalg.norm(obj[:2] - cont[:2])) < 0.06; grasped = bool(ent.is_grasped(d.physics, t.robot))
        if near and (grasped or H["triggered"]): H["triggered"] = True; return True
        return False

    mode = a.release if a.task in GOAL else "latch"
    pol.install(provider, release_mode=mode, release_gate_fn=gate if mode == "oracle_gate" else None)
    pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=pol.config)
    summary = []
    for seed in a.seeds:
        d_ = out / f"seed{seed}"; d_.mkdir(exist_ok=True); snaps, slog = [], []; orig_reset, orig_step = Impl.reset, Impl.step; H["triggered"] = False
        def reset(self, *x, **k):
            r = orig_reset(self, *x, **k); snaps.append([C25.snapshot(self._env)]); return r
        def step(self, action):
            r = orig_step(self, action)
            if self._env is not None and snaps: snaps[-1].append(C25.snapshot(self._env))
            return r
        Impl.reset, Impl.step = reset, step; os_ = pol.select_action
        def sel(batch):
            x = os_(batch); slog.append({"phase": int(pol._v2_grasp_phase), "closed_trig": bool(pol._v2_close_triggered), "grip_norm": float(x[0, 6])}); return x
        pol.select_action = sel; set_seed(seed)
        try: info = eval_policy(env=env, policy=pol, env_preprocessor=env_pre, env_postprocessor=env_post, preprocessor=pre, postprocessor=post, n_episodes=1, max_episodes_rendered=1, videos_dir=d_ / "videos", start_seed=seed)
        finally: Impl.reset, Impl.step = orig_reset, orig_step; pol.select_action = os_
        sn = [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in s.items()} for s in snaps[0]]
        rec = {"task": a.task, "seed": seed, "release": mode, "success": bool(info["per_episode"][0]["success"]), "len": len(slog), "events": pol.log, "sv": {k: (np.asarray(v).tolist() if v is not None and not isinstance(v, (bool, float)) else v) for k, v in pol._sv.__dict__.items()} if pol._sv is not None else None}
        (d_ / "snapshots.json").write_text(json.dumps(sn)); (d_ / "stage_log.json").write_text(json.dumps(slog)); summary.append(rec); print({k: rec[k] for k in ("task", "seed", "release", "success", "len")}, rec["events"][:1], flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    env.close()


if __name__ == "__main__":
    main()
