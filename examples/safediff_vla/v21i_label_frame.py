#!/usr/bin/env python
"""[v2.1 integration gate 1-3] Three grasp quantities in ONE frame (robot base frame = world - (0,-0.4,0.78), metres):
  A  simulator grasp keypoint  = target_entity.get_grasped_keypoints(physics)[0]   (static until grasped)
  B  expert measured EE position at CLOSE = observation ee_state[:3] at the first close-transition step tc (robot frame)
  C  StagePlan-v2 label grasp_xyz = stageplan_v2_labels.labels_v2(action) on the dataset-format action array
(a) SIM replay of the scripted expert in the EXACT dataset recording pipeline (VLABench scripts/trajectory_generation.py + convert_to_lerobot.py):
    action = [robot-frame waypoint[:6], gripper>0.03], state = ee_state in robot frame. 55 seeded episodes/task (seeds 200000+, never eval seeds).
(b) DATASET check on 60 real TRAIN episodes/task: C (stored label) vs B (observation.state at tc) -- same frame by construction of the dataset.
Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/v21i_label_frame.py --task select_poker"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import stageplan_v2_labels as SV  # noqa: E402
from VLABench.envs import load_env  # noqa: E402
from VLABench.tasks import *  # noqa: E402,F401,F403
from VLABench.robots import Franka  # noqa: E402,F401
from lerobot.utils.random_utils import set_seed  # noqa: E402

OUT = Path("outputs/stageplan_v21/x"); BASE = np.array([0, -0.4, 0.78])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--n", type=int, default=55); a = ap.parse_args(); rows = []
    for k in range(a.n * 2):
        if len(rows) >= a.n: break
        seed = 200000 + k; set_seed(seed); env = load_env(a.task, robot="franka", eval=False); env.reset()
        try:
            robot_pos = np.array(env.robot.robot_config["position"]); ent = env.task.entities[env.task.target_entity]; A = np.array(ent.get_grasped_keypoints(env.physics)[0], float) - robot_pos
            obs_all, wps, ok = [], [], False
            for skill in env.get_expert_skill_sequence():
                obs, wp, st, ok = skill(env); obs_all.extend(obs); wps.extend(wp)
                if ok: break
            if not ok: continue
            W = np.array([np.array(w) - np.concatenate([robot_pos, np.zeros(5)]) for w in wps]); act = np.concatenate([W[:, :6], (W[:, -1] > 0.03).astype(float)[:, None]], 1)
            ee = np.array([o["ee_state"][:3] for o in obs_all], float) - robot_pos; lab = SV.labels_v2(act)
            if not lab["valid"] or len(ee) != len(act): continue
            tc, t1 = lab["tc"], lab["t1"]; rows.append({"seed": seed, "task": a.task, "A": A, "B_tc": ee[tc], "B_t1": ee[t1], "C": lab["grasp_xyz"], "tc": tc, "t1": t1, "target": env.task.target_entity, "n": len(act)})
            print(a.task, seed, "A", A.round(4), "B", ee[tc].round(4), "C", lab["grasp_xyz"].round(4), flush=True)
        finally:
            try: env.close()
            except Exception: pass
    pickle.dump(rows, open(OUT / f"label_frame_sim_{a.task}.pkl", "wb"))

if __name__ == "__main__": main()
