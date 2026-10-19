#!/usr/bin/env python
"""G19 STEP 2 closed-loop diagnostic (poker seeds 3000-3009; ONLY run after the fixed-observation gate). The learned decoder is replaced, in PRE_GRASP only, by the NON-LEARNED geometric chunk (g19_geometry: p0 -> target+8cm -> target, constant speed, 30+20 steps):
 --g19-mode log : no intervention (logging only; must reproduce the G5 episodes)
 --g19-mode geo : at every PRE_GRASP replan the decoder's chunk positions are replaced by the geometric chunk toward the planner target the policy itself computed (--model g4d: G4d prediction = the real pipeline; --model oracle: GT anchor, diagnostic upper bound). The decoder's ROTATION channels are kept as-is; the gripper channel is set to OPEN for the whole PRE_GRASP chunk (raw 1.0); close is ONLY the existing reactive-close rule (EE within 3cm of the planner target; one-shot latch; POST_GRASP then uses the unchanged learned decoder).
Per env step: EE, GT keypoint distance (xy/z/3d), phase, whether the chunk was replaced. Everything else (execute_horizon=50, phase logic, reactive close, language, seeds) is the existing G5 pipeline."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, torch

MODE = sys.argv[sys.argv.index("--g19-mode") + 1]; del sys.argv[sys.argv.index("--g19-mode"):sys.argv.index("--g19-mode") + 2]
sys.path.insert(0, str(Path(__file__).parent))
import g5_closed_loop as G5  # noqa: E402
from g19_geometry import geo_chunk  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import robot_base  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW  # noqa: E402

LOG: list[dict] = []; ST = {"ep": -1, "t": 0}; CHUNKS: list[dict] = []; OPEN_RAW = 1.0


@torch.no_grad()
def pre(policy, batch):
    env = G5.ORACLE_ENV["env"]
    if ST["ep"] != G5.ORACLE_ENV["ep"]: ST.update(ep=G5.ORACLE_ENV["ep"], t=0)
    physics = env._env.physics; task = env._env.task; base = robot_base(env); kp = np.asarray(task.entities[task.target_entity].get_grasped_keypoints(physics)[-1], dtype=float); ee_w = np.asarray(task.robot.get_end_effector_pos(physics), dtype=float); d = ee_w - kp
    LOG.append({"ep": ST["ep"], "t": ST["t"], "ee": ee_w - base, "gt": kp - base, "dxy": float(np.linalg.norm(d[:2])), "dz": float(d[2]), "d3": float(np.linalg.norm(d)), "phase": int(policy._v2_grasp_phase), "replaced": False}); ST["t"] += 1


def main():
    orig_inst = G5.install_policy_instrumentation
    def inst(policy, grounder, preproc):
        r = orig_inst(policy, grounder, preproc); pac = policy.plan_action_chunk; mean, std = policy.action_pos_mean, policy.action_pos_std; gm, gs = policy.action_grip_mean, policy.action_grip_std
        def geo_pac(batch):
            actions, metrics = pac(batch)
            if MODE != "geo" or policy._v2_grasp_phase != 0 or policy._v2_last_target_xyz is None: return actions, metrics
            ee = (policy._current_state(batch)[0, :3] * policy.state_pos_std + policy.state_pos_mean).cpu().numpy(); tgt = (policy._v2_last_target_xyz[0] * std + mean).cpu().numpy(); g = geo_chunk(ee, tgt); a = actions.clone()
            a[0, :, :3] = ((torch.as_tensor(g, dtype=a.dtype, device=a.device) - mean) / std); a[0, :, GRIPPER_INDEX_RAW] = ((OPEN_RAW - gm) / gs).to(a.dtype); LOG[-1]["replaced"] = True; CHUNKS.append({"ep": LOG[-1]["ep"], "t": LOG[-1]["t"], "ee": ee, "target": tgt, "first_off": float(np.linalg.norm(g[0] - ee))}); return a, metrics
        policy.plan_action_chunk = geo_pac; return r
    G5.install_policy_instrumentation = inst; G5.PRE["fn"] = pre
    out = Path(sys.argv[sys.argv.index("--out") + 1]); model = sys.argv[sys.argv.index("--model") + 1]
    try: G5.main()
    finally: (out / model).mkdir(parents=True, exist_ok=True); pickle.dump({"log": LOG, "chunks": CHUNKS, "mode": MODE}, open(out / model / "g19_log.pkl", "wb")); print("g19 log steps:", len(LOG), "replaced replans:", len(CHUNKS))


if __name__ == "__main__":
    main()
